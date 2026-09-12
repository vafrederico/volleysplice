import { readFileSync } from "node:fs";
import { parseModelFeedbackText } from "../../prod/src/lib/model-feedback-validation.ts";

const filename = process.argv[2];
if (!filename) throw new Error("Usage: node --experimental-strip-types ios/scripts/validate-feedback.mts <Swift-exported.json>");
const bundle = parseModelFeedbackText(readFileSync(filename, "utf8"));
console.log(JSON.stringify({
  schemaVersion: bundle.schemaVersion,
  runtimeVariant: bundle.source.runtimeVariant,
  featureRows: bundle.features?.rows ?? 0,
  featureColumns: bundle.features?.columns ?? 0,
  inferenceRanges: bundle.initialInference.ranges.length,
  correctedRanges: bundle.corrections.correctedRanges.length,
  ignoredIntervals: bundle.corrections.ignoredIntervals.length,
  finalExportSeconds: bundle.finalExportIntervals.reduce((sum, range) => sum + range.end - range.start, 0),
  validation: "accepted by unchanged production browser parser",
}, null, 2));
