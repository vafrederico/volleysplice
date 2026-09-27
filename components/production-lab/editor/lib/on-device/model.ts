import type { Rally } from "../edit-list";

export class OnDeviceModelError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OnDeviceModelError";
  }
}

export type ProbabilityDecoderConfig = {
  smoothing_seconds: number;
  enter_threshold: number;
  exit_threshold: number;
  min_live_seconds: number;
  bridge_gap_seconds: number;
  short_event_min_seconds: number;
  short_event_threshold: number;
};

export type ServeDecoderConfig = {
  threshold: number;
  minSeparationSeconds: number;
  timeOffsetSeconds: number;
};

export type ServeCompositionConfig = {
  method: "serve-anchor-permissive-live-fallback-v1";
  associationSeconds: number;
  fallbackSeconds: number;
  maxRescueSeconds: number;
  permissiveDecoder: ProbabilityDecoderConfig;
};

export type DeadStateDecoderConfig = {
  deadThreshold: number;
  liveResetThreshold: number;
  minimumLiveSamples: number;
  minimumDeadSamples: number;
  minAfterServeSeconds: number;
  maxAfterServeSeconds: number;
  timeOffsetSeconds: number;
};

export type DeadStateRefinementConfig = {
  method: "v5-noop" | "refine-v5-end";
  endWindowSeconds: number;
};

export type LogisticHead = {
  mean: Float32Array;
  scale: Float32Array;
  weights: Float32Array;
  bias: number;
};

export type OnDeviceModelBundle = {
  schemaVersion: 1;
  analysisFps: number;
  featureNames: readonly string[];
  rally: LogisticHead & { decoder: ProbabilityDecoderConfig };
  serve: LogisticHead & {
    decoder: ServeDecoderConfig;
    composition: ServeCompositionConfig;
  };
  deadState: LogisticHead & {
    decoder: DeadStateDecoderConfig;
    refinement: DeadStateRefinementConfig;
  };
};

export type DecodedInterval = {
  start: number;
  end: number;
  confidence: number;
};

export type ServeDetection = {
  time: number;
  confidence: number;
};

export type DeadStateDetection = {
  time: number;
  confidence: number;
};

export type ProbabilityDecodeResult = {
  intervals: DecodedInterval[];
  smoothed: Float32Array;
};

export type DeadStateRefinementResult = {
  intervals: DecodedInterval[];
  /** One slot per input interval; a missing transition is an exact no-op. */
  transitions: Array<DeadStateDetection | null>;
};

export type OnDeviceInferenceResult = {
  rallies: Rally[];
  intervals: DecodedInterval[];
  primaryIntervals: DecodedInterval[];
  permissiveIntervals: DecodedInterval[];
  serves: ServeDetection[];
  deadStateTransitions: Array<DeadStateDetection | null>;
  probabilities: {
    rally: Float32Array;
    serve: Float32Array;
    deadState: Float32Array;
    smoothedRally: Float32Array;
  };
};

type UnknownRecord = Record<string, unknown>;

function objectValue(value: unknown, label: string): UnknownRecord {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new OnDeviceModelError(`${label} must be an object`);
  }
  return value as UnknownRecord;
}

function finiteNumber(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new OnDeviceModelError(`${label} must be a finite number`);
  }
  return value;
}

function optionalFiniteNumber(
  object: UnknownRecord,
  keys: readonly string[],
  fallback: number,
  label: string,
): number {
  for (const key of keys) {
    if (object[key] !== undefined) return finiteNumber(object[key], label);
  }
  return fallback;
}

function requiredFiniteNumber(
  object: UnknownRecord,
  keys: readonly string[],
  label: string,
): number {
  for (const key of keys) {
    if (object[key] !== undefined) return finiteNumber(object[key], label);
  }
  throw new OnDeviceModelError(`${label} is required`);
}

function positiveInteger(value: unknown, label: string): number {
  const parsed = finiteNumber(value, label);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new OnDeviceModelError(`${label} must be a positive integer`);
  }
  return parsed;
}

function float32Vector(value: unknown, length: number, label: string): Float32Array {
  if (!Array.isArray(value) && !ArrayBuffer.isView(value)) {
    throw new OnDeviceModelError(`${label} must be a numeric array`);
  }
  const source = value as ArrayLike<unknown>;
  if (source.length !== length) {
    throw new OnDeviceModelError(`${label} must contain ${length} values, got ${source.length}`);
  }
  const result = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    result[index] = finiteNumber(source[index], `${label}[${index}]`);
    if (!Number.isFinite(result[index])) {
      throw new OnDeviceModelError(`${label}[${index}] is outside the float32 range`);
    }
  }
  return result;
}

function probabilityDecoder(value: unknown, label: string): ProbabilityDecoderConfig {
  const object = objectValue(value, label);
  const minLiveSeconds = requiredFiniteNumber(
    object,
    ["min_live_seconds", "minLiveSeconds"],
    `${label}.min_live_seconds`,
  );
  const config: ProbabilityDecoderConfig = {
    smoothing_seconds: requiredFiniteNumber(
      object,
      ["smoothing_seconds", "smoothingSeconds"],
      `${label}.smoothing_seconds`,
    ),
    enter_threshold: requiredFiniteNumber(
      object,
      ["enter_threshold", "enterThreshold"],
      `${label}.enter_threshold`,
    ),
    exit_threshold: requiredFiniteNumber(
      object,
      ["exit_threshold", "exitThreshold"],
      `${label}.exit_threshold`,
    ),
    min_live_seconds: minLiveSeconds,
    bridge_gap_seconds: requiredFiniteNumber(
      object,
      ["bridge_gap_seconds", "bridgeGapSeconds"],
      `${label}.bridge_gap_seconds`,
    ),
    // Old artifacts omitted the short-event path. Match DecoderConfig.from_dict.
    short_event_min_seconds: optionalFiniteNumber(
      object,
      ["short_event_min_seconds", "shortEventMinSeconds"],
      minLiveSeconds,
      `${label}.short_event_min_seconds`,
    ),
    short_event_threshold: optionalFiniteNumber(
      object,
      ["short_event_threshold", "shortEventThreshold"],
      1,
      `${label}.short_event_threshold`,
    ),
  };
  validateProbabilityDecoder(config, label);
  return config;
}

function validateProbabilityDecoder(config: ProbabilityDecoderConfig, label: string): void {
  if (
    !Number.isFinite(config.smoothing_seconds) ||
    !Number.isFinite(config.min_live_seconds) ||
    !Number.isFinite(config.bridge_gap_seconds) ||
    !Number.isFinite(config.short_event_min_seconds) ||
    config.smoothing_seconds < 0 ||
    config.min_live_seconds < 0 ||
    config.bridge_gap_seconds < 0 ||
    config.short_event_min_seconds < 0
  ) {
    throw new OnDeviceModelError(`${label} durations must be non-negative`);
  }
  if (
    !Number.isFinite(config.enter_threshold) ||
    !Number.isFinite(config.exit_threshold) ||
    !(config.exit_threshold > 0) ||
    config.exit_threshold > config.enter_threshold ||
    !(config.enter_threshold < 1)
  ) {
    throw new OnDeviceModelError(`${label} thresholds must satisfy 0 < exit <= enter < 1`);
  }
  if (
    !Number.isFinite(config.short_event_threshold) ||
    config.short_event_threshold < config.enter_threshold ||
    config.short_event_threshold > 1
  ) {
    throw new OnDeviceModelError(
      `${label}.short_event_threshold must be between enter_threshold and 1`,
    );
  }
}

function validateDeadStateDecoder(config: DeadStateDecoderConfig): void {
  if (
    !Number.isFinite(config.deadThreshold) ||
    !(config.deadThreshold > 0 && config.deadThreshold < 1)
  ) {
    throw new OnDeviceModelError("deadState.decoder.deadThreshold must be between zero and one");
  }
  if (
    !Number.isFinite(config.liveResetThreshold) ||
    !(config.liveResetThreshold > 0 && config.liveResetThreshold < config.deadThreshold)
  ) {
    throw new OnDeviceModelError(
      "deadState.decoder.liveResetThreshold must be between zero and deadThreshold",
    );
  }
  for (const [label, value] of [
    ["minimumLiveSamples", config.minimumLiveSamples],
    ["minimumDeadSamples", config.minimumDeadSamples],
  ] as const) {
    if (!Number.isInteger(value) || value < 1) {
      throw new OnDeviceModelError(`deadState.decoder.${label} must be a positive integer`);
    }
  }
  if (
    !Number.isFinite(config.minAfterServeSeconds) ||
    !Number.isFinite(config.maxAfterServeSeconds) ||
    config.minAfterServeSeconds < 0 ||
    config.maxAfterServeSeconds < config.minAfterServeSeconds ||
    !Number.isFinite(config.timeOffsetSeconds)
  ) {
    throw new OnDeviceModelError("invalid dead-state decoder window or time offset");
  }
}

function validateDeadStateRefinement(config: DeadStateRefinementConfig): void {
  if (config.method !== "v5-noop" && config.method !== "refine-v5-end") {
    throw new OnDeviceModelError(`unsupported dead-state refinement method: ${String(config.method)}`);
  }
  if (
    !Number.isFinite(config.endWindowSeconds) ||
    config.endWindowSeconds < 0 ||
    (config.method === "refine-v5-end" && config.endWindowSeconds <= 0)
  ) {
    throw new OnDeviceModelError("dead-state refinement requires a valid positive end window");
  }
}

function serveDecoder(value: unknown): ServeDecoderConfig {
  const object = objectValue(value, "serve.decoder");
  const config: ServeDecoderConfig = {
    threshold: requiredFiniteNumber(object, ["threshold"], "serve.decoder.threshold"),
    minSeparationSeconds: requiredFiniteNumber(
      object,
      ["minSeparationSeconds", "min_separation_seconds"],
      "serve.decoder.minSeparationSeconds",
    ),
    timeOffsetSeconds: optionalFiniteNumber(
      object,
      ["timeOffsetSeconds", "time_offset_seconds"],
      0,
      "serve.decoder.timeOffsetSeconds",
    ),
  };
  if (!(config.threshold > 0 && config.threshold < 1)) {
    throw new OnDeviceModelError("serve.decoder.threshold must be between zero and one");
  }
  if (config.minSeparationSeconds < 0) {
    throw new OnDeviceModelError("serve.decoder.minSeparationSeconds must be non-negative");
  }
  return config;
}

function serveComposition(value: unknown): ServeCompositionConfig {
  const object = objectValue(value, "serve.composition");
  if (object.method !== "serve-anchor-permissive-live-fallback-v1") {
    throw new OnDeviceModelError(`unsupported serve composition method: ${String(object.method)}`);
  }
  const config: ServeCompositionConfig = {
    method: "serve-anchor-permissive-live-fallback-v1",
    associationSeconds: requiredFiniteNumber(
      object,
      ["associationSeconds", "association_seconds"],
      "serve.composition.associationSeconds",
    ),
    fallbackSeconds: requiredFiniteNumber(
      object,
      ["fallbackSeconds", "fallback_seconds"],
      "serve.composition.fallbackSeconds",
    ),
    maxRescueSeconds: requiredFiniteNumber(
      object,
      ["maxRescueSeconds", "max_rescue_seconds"],
      "serve.composition.maxRescueSeconds",
    ),
    permissiveDecoder: probabilityDecoder(
      object.permissiveDecoder ?? object.permissive_decoder,
      "serve.composition.permissiveDecoder",
    ),
  };
  if (
    config.associationSeconds < 0 ||
    config.fallbackSeconds < 0 ||
    !(config.maxRescueSeconds > 0)
  ) {
    throw new OnDeviceModelError(
      "serve composition durations must be non-negative and maxRescueSeconds must be positive",
    );
  }
  return config;
}

function deadStateDecoder(value: unknown): DeadStateDecoderConfig {
  const object = objectValue(value, "deadState.decoder");
  const config: DeadStateDecoderConfig = {
    deadThreshold: requiredFiniteNumber(
      object,
      ["deadThreshold", "dead_threshold"],
      "deadState.decoder.deadThreshold",
    ),
    liveResetThreshold: requiredFiniteNumber(
      object,
      ["liveResetThreshold", "live_reset_threshold"],
      "deadState.decoder.liveResetThreshold",
    ),
    minimumLiveSamples: positiveInteger(
      object.minimumLiveSamples ?? object.minimum_live_samples ?? 1,
      "deadState.decoder.minimumLiveSamples",
    ),
    minimumDeadSamples: positiveInteger(
      object.minimumDeadSamples ?? object.minimum_dead_samples ?? 2,
      "deadState.decoder.minimumDeadSamples",
    ),
    minAfterServeSeconds: optionalFiniteNumber(
      object,
      ["minAfterServeSeconds", "min_after_serve_seconds"],
      0.25,
      "deadState.decoder.minAfterServeSeconds",
    ),
    maxAfterServeSeconds: optionalFiniteNumber(
      object,
      ["maxAfterServeSeconds", "max_after_serve_seconds"],
      4,
      "deadState.decoder.maxAfterServeSeconds",
    ),
    timeOffsetSeconds: optionalFiniteNumber(
      object,
      ["timeOffsetSeconds", "time_offset_seconds"],
      0,
      "deadState.decoder.timeOffsetSeconds",
    ),
  };
  validateDeadStateDecoder(config);
  return config;
}

function deadStateRefinement(value: unknown): DeadStateRefinementConfig {
  const object = objectValue(value, "deadState.refinement");
  const method = object.method;
  if (method !== "v5-noop" && method !== "refine-v5-end") {
    throw new OnDeviceModelError(`unsupported dead-state refinement method: ${String(method)}`);
  }
  const endWindowSeconds = optionalFiniteNumber(
    object,
    ["endWindowSeconds", "end_window_seconds"],
    1,
    "deadState.refinement.endWindowSeconds",
  );
  const config: DeadStateRefinementConfig = { method, endWindowSeconds };
  validateDeadStateRefinement(config);
  return config;
}

function logisticHead(value: unknown, dimensions: number, label: string): LogisticHead {
  const object = objectValue(value, label);
  const result: LogisticHead = {
    mean: float32Vector(object.mean, dimensions, `${label}.mean`),
    scale: float32Vector(object.scale, dimensions, `${label}.scale`),
    weights: float32Vector(object.weights, dimensions, `${label}.weights`),
    bias: finiteNumber(object.bias, `${label}.bias`),
  };
  for (let index = 0; index < dimensions; index += 1) {
    if (!(result.scale[index] > 0)) {
      throw new OnDeviceModelError(`${label}.scale[${index}] must be positive`);
    }
  }
  return result;
}

/** Parse and validate the build-time JSON representation of the three model heads. */
export function loadOnDeviceModelBundle(value: unknown): OnDeviceModelBundle {
  const object = objectValue(value, "model bundle");
  if (object.schemaVersion !== 1) {
    throw new OnDeviceModelError(`unsupported model bundle schemaVersion: ${String(object.schemaVersion)}`);
  }
  const analysisFps = finiteNumber(object.analysisFps, "model bundle analysisFps");
  if (!(analysisFps > 0)) {
    throw new OnDeviceModelError("model bundle analysisFps must be positive");
  }
  if (
    !Array.isArray(object.featureNames) ||
    object.featureNames.length === 0 ||
    object.featureNames.some((name) => typeof name !== "string" || name.length === 0)
  ) {
    throw new OnDeviceModelError("model bundle featureNames must be a non-empty string array");
  }
  const featureNames = [...(object.featureNames as string[])];
  if (new Set(featureNames).size !== featureNames.length) {
    throw new OnDeviceModelError("model bundle featureNames must be unique");
  }

  const rallyObject = objectValue(object.rally, "rally");
  const serveObject = objectValue(object.serve, "serve");
  const deadObject = objectValue(object.deadState, "deadState");
  const rally = {
    ...logisticHead(rallyObject, featureNames.length, "rally"),
    decoder: probabilityDecoder(rallyObject.decoder, "rally.decoder"),
  };
  const serve = {
    ...logisticHead(serveObject, featureNames.length, "serve"),
    decoder: serveDecoder(serveObject.decoder),
    composition: serveComposition(serveObject.composition),
  };
  const deadState = {
    ...logisticHead(deadObject, featureNames.length, "deadState"),
    decoder: deadStateDecoder(deadObject.decoder),
    refinement: deadStateRefinement(deadObject.refinement),
  };
  return { schemaVersion: 1, analysisFps, featureNames, rally, serve, deadState };
}

/**
 * Apply float32 normalization and a clipped weighted-logistic head to a flat,
 * row-major feature matrix. This deliberately stays on CPU: at 520 columns the
 * WebGPU dispatch overhead is larger than the classifier work.
 */
export function predictLogistic(
  head: LogisticHead,
  values: ArrayLike<number>,
  rowCount?: number,
): Float32Array {
  const dimensions = head.weights.length;
  if (dimensions === 0 || head.mean.length !== dimensions || head.scale.length !== dimensions) {
    throw new OnDeviceModelError("model head parameter dimensions do not agree");
  }
  const rows = rowCount ?? values.length / dimensions;
  if (!Number.isInteger(rows) || rows < 0 || values.length !== rows * dimensions) {
    throw new OnDeviceModelError(
      `model expects a flat (*, ${dimensions}) matrix, got ${values.length} values`,
    );
  }
  const probabilities = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * dimensions;
    let logit = Math.fround(head.bias);
    for (let column = 0; column < dimensions; column += 1) {
      const value = values[offset + column];
      if (!Number.isFinite(value)) {
        throw new OnDeviceModelError(`feature matrix contains a non-finite value at row ${row}, column ${column}`);
      }
      const normalized = Math.fround(
        Math.fround(Math.fround(value) - head.mean[column]) / head.scale[column],
      );
      logit = Math.fround(logit + Math.fround(normalized * head.weights[column]));
    }
    const clipped = Math.max(-30, Math.min(30, logit));
    probabilities[row] = 1 / (1 + Math.exp(-clipped));
  }
  return probabilities;
}

function pythonRoundNonNegative(value: number): number {
  const floor = Math.floor(value);
  const fraction = value - floor;
  if (fraction < 0.5) return floor;
  if (fraction > 0.5) return floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

function smoothProbabilities(
  probabilities: ArrayLike<number>,
  windowSamples: number,
): Float32Array {
  const length = probabilities.length;
  const result = new Float32Array(length);
  if (length === 0 || windowSamples <= 1) {
    for (let index = 0; index < length; index += 1) result[index] = probabilities[index];
    return result;
  }
  const window = Math.min(windowSamples, length);
  const leftPadding = Math.floor(window / 2);
  const scale = Math.fround(1 / window);
  for (let output = 0; output < length; output += 1) {
    let sum = Math.fround(0);
    for (let kernel = 0; kernel < window; kernel += 1) {
      const source = Math.max(0, Math.min(length - 1, output + kernel - leftPadding));
      sum = Math.fround(sum + Math.fround(probabilities[source] * scale));
    }
    result[output] = sum;
  }
  return result;
}

function runs(mask: Uint8Array, value: boolean): Array<[number, number]> {
  const expected = value ? 1 : 0;
  const result: Array<[number, number]> = [];
  let start: number | null = null;
  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === expected && start === null) start = index;
    if (mask[index] !== expected && start !== null) {
      result.push([start, index]);
      start = null;
    }
  }
  if (start !== null) result.push([start, mask.length]);
  return result;
}

function meanFloat32(values: ArrayLike<number>, start: number, end: number): number {
  let sum = Math.fround(0);
  for (let index = start; index < end; index += 1) {
    sum = Math.fround(sum + values[index]);
  }
  return Math.fround(sum / (end - start));
}

function validateTimes(times: ArrayLike<number>, label: string): void {
  for (let index = 0; index < times.length; index += 1) {
    if (!Number.isFinite(times[index])) {
      throw new OnDeviceModelError(`${label} times must be finite`);
    }
    if (index > 0 && !(times[index] > times[index - 1])) {
      throw new OnDeviceModelError(`${label} times must be strictly increasing`);
    }
  }
}

function validateProbabilities(probabilities: ArrayLike<number>, label: string): void {
  for (let index = 0; index < probabilities.length; index += 1) {
    const probability = probabilities[index];
    if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
      throw new OnDeviceModelError(`${label} probabilities must be finite and between zero and one`);
    }
  }
}

/** Port of analysis.decoder.decode_probabilities. */
export function decodeProbabilities(
  times: ArrayLike<number>,
  probabilities: ArrayLike<number>,
  duration: number,
  config: ProbabilityDecoderConfig,
  analysisFps: number,
): ProbabilityDecodeResult {
  validateProbabilityDecoder(config, "decoder");
  if (times.length !== probabilities.length) {
    throw new OnDeviceModelError("times and probabilities must be aligned arrays");
  }
  validateTimes(times, "rally decode");
  validateProbabilities(probabilities, "rally decode");
  if (!Number.isFinite(duration) || duration < 0 || !Number.isFinite(analysisFps) || analysisFps <= 0) {
    throw new OnDeviceModelError("decode duration must be non-negative and analysisFps must be positive");
  }
  if (times.length === 0) {
    return { intervals: [], smoothed: new Float32Array(probabilities) };
  }

  const smoothingSamples = Math.max(
    1,
    pythonRoundNonNegative(config.smoothing_seconds * analysisFps),
  );
  const smoothed = smoothProbabilities(probabilities, smoothingSamples);
  const mask = new Uint8Array(smoothed.length);
  let live = false;
  for (let index = 0; index < smoothed.length; index += 1) {
    if (!live && smoothed[index] >= config.enter_threshold) live = true;
    else if (live && smoothed[index] < config.exit_threshold) live = false;
    mask[index] = live ? 1 : 0;
  }

  const bridgeSamples = Math.max(
    0,
    pythonRoundNonNegative(config.bridge_gap_seconds * analysisFps),
  );
  if (bridgeSamples > 0) {
    for (const [start, end] of runs(mask, false)) {
      if (start > 0 && end < mask.length && end - start <= bridgeSamples) {
        mask.fill(1, start, end);
      }
    }
  }

  const minimumLiveSamples = Math.max(
    1,
    pythonRoundNonNegative(config.min_live_seconds * analysisFps),
  );
  const shortEventMinimumSamples = Math.max(
    1,
    pythonRoundNonNegative(config.short_event_min_seconds * analysisFps),
  );
  if (minimumLiveSamples > 1) {
    for (const [start, end] of runs(mask, true)) {
      let peak = -Infinity;
      for (let index = start; index < end; index += 1) peak = Math.max(peak, smoothed[index]);
      const keepShort =
        end - start >= shortEventMinimumSamples && peak >= config.short_event_threshold;
      if (end - start < minimumLiveSamples && !keepShort) mask.fill(0, start, end);
    }
  }

  const sampleWidth = 1 / analysisFps;
  const intervals: DecodedInterval[] = [];
  for (const [startIndex, endIndex] of runs(mask, true)) {
    const start = Math.max(0, times[startIndex] - sampleWidth / 2);
    const end = Math.min(duration, times[endIndex - 1] + sampleWidth / 2);
    if (end <= start) continue;
    intervals.push({
      start,
      end,
      confidence: meanFloat32(smoothed, startIndex, endIndex),
    });
  }
  return { intervals, smoothed };
}

/** Port of analysis.serve.decode_serve_probabilities. */
export function decodeServeProbabilities(
  times: ArrayLike<number>,
  probabilities: ArrayLike<number>,
  config: ServeDecoderConfig,
  duration?: number,
): ServeDetection[] {
  if (times.length !== probabilities.length) {
    throw new OnDeviceModelError("serve times and probabilities must be aligned arrays");
  }
  validateTimes(times, "serve decode");
  validateProbabilities(probabilities, "serve decode");
  if (
    !(config.threshold > 0 && config.threshold < 1) ||
    !Number.isFinite(config.minSeparationSeconds) ||
    config.minSeparationSeconds < 0 ||
    !Number.isFinite(config.timeOffsetSeconds)
  ) {
    throw new OnDeviceModelError("invalid serve decoder configuration");
  }
  if (duration !== undefined && (!Number.isFinite(duration) || duration <= 0)) {
    throw new OnDeviceModelError("serve decode duration must be positive when supplied");
  }

  const candidates: ServeDetection[] = [];
  let activeStart: number | null = null;
  for (let index = 0; index < probabilities.length; index += 1) {
    const active = probabilities[index] >= config.threshold;
    if (active && activeStart === null) activeStart = index;
    const closes = activeStart !== null && (!active || index === probabilities.length - 1);
    if (!closes || activeStart === null) continue;
    const end = active ? index + 1 : index;
    let peakIndex = activeStart;
    for (let candidate = activeStart + 1; candidate < end; candidate += 1) {
      if (probabilities[candidate] > probabilities[peakIndex]) peakIndex = candidate;
    }
    let contactTime = times[peakIndex] + config.timeOffsetSeconds;
    if (duration !== undefined) contactTime = Math.min(duration, Math.max(0, contactTime));
    candidates.push({ time: contactTime, confidence: probabilities[peakIndex] });
    activeStart = null;
  }

  candidates.sort((left, right) => right.confidence - left.confidence || left.time - right.time);
  const retained: ServeDetection[] = [];
  for (const candidate of candidates) {
    if (
      retained.every(
        (previous) =>
          Math.abs(candidate.time - previous.time) >= config.minSeparationSeconds,
      )
    ) {
      retained.push(candidate);
    }
  }
  return retained.sort((left, right) => left.time - right.time);
}

/** Port of analysis.serve.compose_serve_anchored_intervals. */
export function composeServeAnchoredIntervals(
  primary: readonly DecodedInterval[],
  permissive: readonly DecodedInterval[],
  serves: readonly ServeDetection[],
  duration: number,
  config: ServeCompositionConfig,
  sampleSeconds: number,
): DecodedInterval[] {
  if (!Number.isFinite(duration) || duration <= 0 || !Number.isFinite(sampleSeconds) || sampleSeconds <= 0) {
    throw new OnDeviceModelError("composition duration and sample width must be positive");
  }
  const rows: Array<[number, number, number]> = primary.map((interval) => [
    interval.start,
    interval.end,
    interval.confidence,
  ]);
  const unused: ServeDetection[] = [];
  for (const serve of serves) {
    const associated: number[] = [];
    for (let index = 0; index < rows.length; index += 1) {
      if (rows[index][0] - config.associationSeconds <= serve.time && serve.time < rows[index][1]) {
        associated.push(index);
      }
    }
    if (associated.length === 0) {
      unused.push(serve);
      continue;
    }
    let selectedIndex = associated[0];
    for (const candidate of associated.slice(1)) {
      if (Math.abs(rows[candidate][0] - serve.time) < Math.abs(rows[selectedIndex][0] - serve.time)) {
        selectedIndex = candidate;
      }
    }
    if (serve.time < rows[selectedIndex][0]) {
      rows[selectedIndex][0] = serve.time;
      rows[selectedIndex][2] = Math.max(rows[selectedIndex][2], serve.confidence);
    }
  }

  for (const serve of unused) {
    const eligible = permissive.filter(
      (interval) =>
        interval.end >= serve.time - config.associationSeconds &&
        interval.start <= serve.time + config.associationSeconds &&
        interval.end - interval.start <= config.maxRescueSeconds,
    );
    if (eligible.length > 0) {
      let selected = eligible[0];
      for (const candidate of eligible.slice(1)) {
        const candidateDistance = Math.min(
          Math.abs(candidate.start - serve.time),
          Math.abs(candidate.end - serve.time),
        );
        const selectedDistance = Math.min(
          Math.abs(selected.start - serve.time),
          Math.abs(selected.end - serve.time),
        );
        if (candidateDistance < selectedDistance) selected = candidate;
      }
      rows.push([
        serve.time,
        Math.max(serve.time + sampleSeconds, selected.end),
        Math.max(serve.confidence, selected.confidence),
      ]);
    } else if (config.fallbackSeconds > 0) {
      rows.push([serve.time, serve.time + config.fallbackSeconds, serve.confidence]);
    }
  }

  const ordered = rows
    .filter(([start, end]) => end > start && start < duration && end > 0)
    .map<[number, number, number]>(([start, end, confidence]) => [
      Math.max(0, start),
      Math.min(duration, end),
      confidence,
    ])
    .sort(
      (left, right) =>
        left[0] - right[0] || left[1] - right[1] || left[2] - right[2],
    );
  const merged: Array<[number, number, number]> = [];
  for (const [start, end, confidence] of ordered) {
    const previous = merged[merged.length - 1];
    if (previous && start < previous[1]) {
      previous[1] = Math.max(previous[1], end);
      previous[2] = Math.max(previous[2], confidence);
    } else {
      merged.push([start, end, confidence]);
    }
  }
  return merged.map(([start, end, confidence]) => ({ start, end, confidence }));
}

/** Port of analysis.dead_state.decode_dead_state_after_serve. */
export function decodeDeadStateAfterServe(
  times: ArrayLike<number>,
  probabilities: ArrayLike<number>,
  serveTime: number,
  config: DeadStateDecoderConfig,
  options: { nextServeTime?: number; duration?: number } = {},
): DeadStateDetection | null {
  validateDeadStateDecoder(config);
  if (times.length !== probabilities.length) {
    throw new OnDeviceModelError("dead-state times and probabilities must be aligned arrays");
  }
  validateTimes(times, "dead-state decode");
  validateProbabilities(probabilities, "dead-state decode");
  if (!Number.isFinite(serveTime)) throw new OnDeviceModelError("dead-state anchor must be finite");
  if (
    options.nextServeTime !== undefined &&
    (!Number.isFinite(options.nextServeTime) || options.nextServeTime <= serveTime)
  ) {
    throw new OnDeviceModelError("dead-state next anchor must be finite and after the anchor");
  }
  if (options.duration !== undefined && (!Number.isFinite(options.duration) || options.duration <= 0)) {
    throw new OnDeviceModelError("dead-state duration must be positive when supplied");
  }

  const detectionLower = serveTime + config.minAfterServeSeconds;
  const detectionUpper = serveTime + config.maxAfterServeSeconds;
  let liveRun = 0;
  let armed = false;
  let deadRunStart: number | null = null;
  let deadRunLength = 0;
  for (let index = 0; index < times.length; index += 1) {
    const time = times[index];
    const probability = probabilities[index];
    if (time < serveTime) continue;
    if (time > detectionUpper) break;
    if (options.nextServeTime !== undefined && time >= options.nextServeTime) break;

    if (!armed) {
      if (probability <= config.liveResetThreshold) {
        liveRun += 1;
        if (liveRun >= config.minimumLiveSamples) armed = true;
      } else {
        liveRun = 0;
      }
    }
    if (!armed || time < detectionLower) {
      deadRunStart = null;
      deadRunLength = 0;
      continue;
    }
    if (probability >= config.deadThreshold) {
      if (deadRunStart === null) deadRunStart = index;
      deadRunLength += 1;
    } else {
      deadRunStart = null;
      deadRunLength = 0;
    }
    if (deadRunLength < config.minimumDeadSamples || deadRunStart === null) continue;

    let confidence = Infinity;
    for (let runIndex = deadRunStart; runIndex <= index; runIndex += 1) {
      confidence = Math.min(confidence, probabilities[runIndex]);
    }
    let detectionTime = times[deadRunStart] + config.timeOffsetSeconds;
    if (options.duration !== undefined) {
      detectionTime = Math.min(options.duration, Math.max(0, detectionTime));
    }
    return { time: detectionTime, confidence };
  }
  return null;
}

/** Refine existing v5 ends only; this intentionally cannot create a missing interval. */
export function refineDeadStateEnds(
  times: ArrayLike<number>,
  probabilities: ArrayLike<number>,
  intervals: readonly DecodedInterval[],
  duration: number,
  sampleSeconds: number,
  decoder: DeadStateDecoderConfig,
  refinement: DeadStateRefinementConfig,
): DeadStateRefinementResult {
  validateDeadStateDecoder(decoder);
  validateDeadStateRefinement(refinement);
  if (refinement.method === "v5-noop") {
    return { intervals: intervals.map((interval) => ({ ...interval })), transitions: [] };
  }
  const transitions: Array<DeadStateDetection | null> = [];
  for (let index = 0; index < intervals.length; index += 1) {
    const interval = intervals[index];
    const nextStart = intervals[index + 1]?.start;
    const anchor = Math.max(0, interval.end - refinement.endWindowSeconds);
    const localDecoder: DeadStateDecoderConfig = {
      ...decoder,
      minAfterServeSeconds: 0,
      maxAfterServeSeconds: interval.end + refinement.endWindowSeconds - anchor,
    };
    const detection = decodeDeadStateAfterServe(times, probabilities, anchor, localDecoder, {
      nextServeTime: nextStart !== undefined && nextStart > anchor ? nextStart : undefined,
      duration,
    });
    transitions.push(
      detection === null
        ? null
        : {
            time: Math.min(
              duration,
              interval.end + refinement.endWindowSeconds,
              Math.max(interval.end - refinement.endWindowSeconds, detection.time),
            ),
            confidence: detection.confidence,
          },
    );
  }

  const refined = intervals.map((interval, index) => {
    const transition = transitions[index];
    if (transition === null) return { ...interval };
    const nextStart = intervals[index + 1]?.start ?? duration;
    const refinedEnd = Math.min(duration, nextStart, transition.time);
    if (refinedEnd < interval.start + sampleSeconds - 1e-9) return { ...interval };
    return {
      start: interval.start,
      end: refinedEnd,
      confidence: Math.max(interval.confidence, transition.confidence),
    };
  });
  return { intervals: refined, transitions };
}

function effectiveAnalysisFps(times: ArrayLike<number>, fallback: number): number {
  if (times.length <= 1) return 1;
  const differences = new Array<number>(times.length - 1);
  for (let index = 1; index < times.length; index += 1) {
    differences[index - 1] = times[index] - times[index - 1];
  }
  differences.sort((left, right) => left - right);
  const middle = Math.floor(differences.length / 2);
  const median =
    differences.length % 2 === 1
      ? differences[middle]
      : (differences[middle - 1] + differences[middle]) / 2;
  return Number.isFinite(median) && median > 0 ? 1 / median : fallback;
}

/** Execute the exact rally -> serve composition -> dead-end refinement chain. */
export function runOnDeviceModel(
  bundle: OnDeviceModelBundle,
  times: ArrayLike<number>,
  contextualFeatures: ArrayLike<number>,
  duration: number,
): OnDeviceInferenceResult {
  validateTimes(times, "inference");
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new OnDeviceModelError("inference duration must be positive");
  }
  const rallyProbabilities = predictLogistic(bundle.rally, contextualFeatures, times.length);
  const serveProbabilities = predictLogistic(bundle.serve, contextualFeatures, times.length);
  const deadStateProbabilities = predictLogistic(
    bundle.deadState,
    contextualFeatures,
    times.length,
  );
  const fps = effectiveAnalysisFps(times, bundle.analysisFps);
  const primary = decodeProbabilities(
    times,
    rallyProbabilities,
    duration,
    bundle.rally.decoder,
    fps,
  );
  const permissive = decodeProbabilities(
    times,
    rallyProbabilities,
    duration,
    bundle.serve.composition.permissiveDecoder,
    fps,
  );
  const serves = decodeServeProbabilities(
    times,
    serveProbabilities,
    bundle.serve.decoder,
    duration,
  );
  const composed = composeServeAnchoredIntervals(
    primary.intervals,
    permissive.intervals,
    serves,
    duration,
    bundle.serve.composition,
    1 / fps,
  );
  const refined = refineDeadStateEnds(
    times,
    deadStateProbabilities,
    composed,
    duration,
    1 / fps,
    bundle.deadState.decoder,
    bundle.deadState.refinement,
  );
  const rallies: Rally[] = refined.intervals.map((interval, index) => ({
    id: `R${String(index + 1).padStart(3, "0")}`,
    start: Math.max(0, interval.start),
    end: Math.min(duration, interval.end),
    confidence: interval.confidence,
    included: true,
  }));
  return {
    rallies,
    intervals: refined.intervals,
    primaryIntervals: primary.intervals,
    permissiveIntervals: permissive.intervals,
    serves,
    deadStateTransitions: refined.transitions,
    probabilities: {
      rally: rallyProbabilities,
      serve: serveProbabilities,
      deadState: deadStateProbabilities,
      smoothedRally: primary.smoothed,
    },
  };
}
