import { servingSideFeatureNames, tiedPercentileRanks } from "./serving-side-model.ts";

/** 237 recording ranks followed by 12 unranked, phase-specific quality values. */
export function neuralServingFeatures(raw: Float64Array, rows: number): Float32Array {
  if (!Number.isSafeInteger(rows) || rows < 0 || raw.length !== rows * 237 || !raw.every(Number.isFinite)) {
    throw new Error("Invalid neural serving-side specialist features.");
  }
  const ranks = tiedPercentileRanks(raw, rows, 237), names = servingSideFeatureNames();
  const qualityColumns = ["launch", "early", "late"].flatMap(phase =>
    ["energyMean", "activeFraction", "smallComponentEnergyFraction", "flowY"].map(name => names.indexOf(`flight:${phase}:${name}`)));
  if (qualityColumns.some(index => index < 0)) throw new Error("The serving-side quality signature changed.");
  const features = new Float32Array(rows * 249);
  for (let row = 0; row < rows; row++) {
    features.set(ranks.subarray(row * 237, (row + 1) * 237), row * 249);
    for (let column = 0; column < 12; column++) {
      const value = raw[row * 237 + qualityColumns[column]];
      features[row * 249 + 237 + column] = column % 4 === 3 ? Math.abs(value) : value;
    }
  }
  return features;
}
