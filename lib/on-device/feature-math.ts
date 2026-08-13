import {
  ABSOLUTE_FEATURE_NAMES,
  CONTEXT_OFFSETS_SECONDS,
} from "./feature-schema.ts";

export function mean(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  let total = 0;
  for (let index = 0; index < values.length; index += 1) total += values[index];
  return total / values.length;
}

export function standardDeviation(values: ArrayLike<number>, average = mean(values)): number {
  if (values.length === 0) return 0;
  let sumSquares = 0;
  for (let index = 0; index < values.length; index += 1) {
    const delta = values[index] - average;
    sumSquares += delta * delta;
  }
  return Math.sqrt(sumSquares / values.length);
}

export function quantile(values: ArrayLike<number>, percentile: number): number {
  if (values.length === 0) return 0;
  const sorted = Float64Array.from(values).sort();
  const position = (sorted.length - 1) * Math.min(1, Math.max(0, percentile));
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

export function rollingMean(
  values: ArrayLike<number>,
  windowSamples: number,
  future: boolean,
): Float32Array {
  const result = new Float32Array(values.length);
  if (values.length === 0) return result;
  const window = Math.max(1, Math.trunc(windowSamples));
  const cumulative = new Float64Array(values.length + 1);
  for (let index = 0; index < values.length; index += 1) {
    cumulative[index + 1] = cumulative[index] + values[index];
  }
  for (let index = 0; index < values.length; index += 1) {
    const start = future ? index : Math.max(0, index - window + 1);
    const end = future ? Math.min(values.length, index + window) : index + 1;
    result[index] = (cumulative[end] - cumulative[start]) / (end - start);
  }
  return result;
}

export function percentileRanks(
  values: Float32Array,
  rows: number,
  columns: number,
): Float32Array {
  if (values.length !== rows * columns) {
    throw new Error("Feature matrix dimensions do not match its storage.");
  }
  if (rows === 0) return new Float32Array(values);
  if (rows === 1) return new Float32Array(columns).fill(0.5);

  const result = new Float32Array(values.length);
  const order = new Array<number>(rows);
  const denominator = rows - 1;
  for (let column = 0; column < columns; column += 1) {
    for (let row = 0; row < rows; row += 1) order[row] = row;
    order.sort((left, right) => values[left * columns + column] - values[right * columns + column]);
    let start = 0;
    while (start < rows) {
      const value = values[order[start] * columns + column];
      let end = start + 1;
      while (end < rows && values[order[end] * columns + column] === value) end += 1;
      const midrank = (start + end - 1) / 2 / denominator;
      for (let index = start; index < end; index += 1) {
        result[order[index] * columns + column] = midrank;
      }
      start = end;
    }
  }
  return result;
}

function lowerBound(values: Float64Array, target: number): number {
  let lower = 0;
  let upper = values.length;
  while (lower < upper) {
    const middle = (lower + upper) >>> 1;
    if (values[middle] < target) lower = middle + 1;
    else upper = middle;
  }
  return lower;
}

function nearestIndex(times: Float64Array, target: number): number {
  if (target <= times[0]) return 0;
  if (target >= times[times.length - 1]) return times.length - 1;
  const right = Math.min(times.length - 1, lowerBound(times, target));
  const left = Math.max(0, right - 1);
  return Math.abs(times[left] - target) <= Math.abs(times[right] - target)
    ? left
    : right;
}

export function contextualizeFeatures(
  times: Float64Array,
  base: Float32Array,
  featureNames: readonly string[],
): { values: Float32Array; names: string[] } {
  const rows = times.length;
  const columns = featureNames.length;
  if (rows === 0 || base.length !== rows * columns) {
    throw new Error("Context input must be a non-empty aligned feature matrix.");
  }
  const ranked = percentileRanks(base, rows, columns);
  for (let column = 0; column < columns; column += 1) {
    if (!ABSOLUTE_FEATURE_NAMES.has(featureNames[column])) continue;
    for (let row = 0; row < rows; row += 1) {
      ranked[row * columns + column] = base[row * columns + column];
    }
  }

  const blockColumns = columns * CONTEXT_OFFSETS_SECONDS.length;
  const output = new Float32Array(rows * blockColumns);
  const names: string[] = [];
  for (let block = 0; block < CONTEXT_OFFSETS_SECONDS.length; block += 1) {
    const offset = CONTEXT_OFFSETS_SECONDS[block];
    for (const name of featureNames) names.push(`t${offset >= 0 ? "+" : ""}${offset}s/${name}`);
    for (let row = 0; row < rows; row += 1) {
      const source = nearestIndex(times, times[row] + offset);
      const targetStart = row * blockColumns + block * columns;
      output.set(ranked.subarray(source * columns, (source + 1) * columns), targetStart);
    }
  }
  return { values: output, names };
}

export function medianPositiveDelta(times: Float64Array): number {
  if (times.length < 2) return 0.25;
  const deltas: number[] = [];
  for (let index = 1; index < times.length; index += 1) {
    const delta = times[index] - times[index - 1];
    if (Number.isFinite(delta) && delta > 0) deltas.push(delta);
  }
  return deltas.length ? quantile(deltas, 0.5) : 0.25;
}
