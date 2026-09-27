import type { LabConfiguration } from "./production-editor-lab.ts";
import { parseServingPredictions, servingDecisionExplanation } from "./labeling-serving.ts";

export function servingPopulation(configuration: LabConfiguration) {
  return (configuration.draftEvents ?? configuration.events).map(({ id, start, end, agreement }) =>
    ({ id, start, end, ...(agreement ? { agreement } : {}) }));
}

/** Reject stale geometry rather than assigning another variant's serving-side result. */
export function attachLabServing(configuration: LabConfiguration, value: unknown, duration: number): LabConfiguration {
  const receipt = value as { population?: unknown; output?: unknown };
  if (!receipt || JSON.stringify(receipt.population) !== JSON.stringify(servingPopulation(configuration))) {
    throw new Error("Serving-side results do not match this model's rally population");
  }
  const predictions = parseServingPredictions(receipt.output, duration);
  const population = servingPopulation(configuration);
  if (predictions.length !== population.length || population.some(event =>
    !predictions.some(row => row.id === event.id && Math.abs(row.anchor - event.start) < 1e-6))) {
    throw new Error("Serving-side results do not cover this model's rally anchors");
  }
  const events = (rows: LabConfiguration["events"]) => rows.map(event => {
    const prediction = predictions.find(row => row.id === event.id);
    if (!prediction) return event;
    const { serve: _old, ...rest } = event;
    return { ...rest, ...(prediction.verdict === "not-serve" ? {} : { serve: {
      time: prediction.anchor, side: prediction.verdict === "review" ? "review" as const : prediction.side,
      confidence: prediction.side === "near" ? prediction.nearProbability : 1 - prediction.nearProbability,
      reason: servingDecisionExplanation(prediction),
    } }) };
  });
  return { ...configuration, events: events(configuration.events),
    ...(configuration.draftEvents ? { draftEvents: events(configuration.draftEvents) } : {}), servingPredictions: predictions };
}
