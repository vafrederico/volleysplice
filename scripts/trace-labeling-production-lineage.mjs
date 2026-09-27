// Reconstruct the production decoder/composition chain from saved probabilities.
// No labels enter this process; checked-in production functions are the oracle.
import fs from "node:fs";

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const model = await import(input.repoUrl + "/prod/src/lib/on-device/model.ts");
const ensemble = await import(input.repoUrl + "/prod/src/lib/on-device/ensemble.ts");
const require = (condition, message) => { if (!condition) throw new Error(message); };
const exact = (actual, expected, message) => require(JSON.stringify(actual) === JSON.stringify(expected), message);
const clean = row => ({ start: row.start, end: row.end, confidence: row.confidence });
const overlap = (a, b) => a.start < b.end && b.start < a.end;
const times = Float64Array.from(input.times);
const differences = Array.from(times.slice(1), (value, index) => value - times[index]).sort((a, b) => a - b);
const middle = Math.floor(differences.length / 2);
const sampleSeconds = differences.length % 2 ? differences[middle] : (differences[middle - 1] + differences[middle]) / 2;
const fps = 1 / sampleSeconds;

function traceComposition(primary, permissive, serves, duration, config) {
  const rows = primary.map((interval, index) => ({ ...interval, origins: [{
    kind: "primary-rally", primaryId: `primary:${index + 1}`, originalInterval: clean(interval), serveActions: [],
  }] }));
  const unused = [];
  const actions = [];
  for (const [index, detection] of serves.entries()) {
    const serve = { id: `serve:${index + 1}`, ...detection };
    const associated = rows.filter(row => row.start - config.associationSeconds <= serve.time && serve.time < row.end);
    if (!associated.length) { unused.push(serve); continue; }
    let selected = associated[0];
    for (const candidate of associated.slice(1)) {
      if (Math.abs(candidate.start - serve.time) < Math.abs(selected.start - serve.time)) selected = candidate;
    }
    const action = { serve, kind: serve.time < selected.start ? "existing-rally-start-refinement" : "existing-rally-no-boundary-change",
      primaryId: selected.origins[0].primaryId, startBefore: selected.start, startAfter: Math.min(selected.start, serve.time) };
    actions.push(action);
    selected.origins[0].serveActions.push(action);
    if (serve.time < selected.start) {
      selected.start = serve.time;
      selected.confidence = Math.max(selected.confidence, serve.confidence);
    }
  }
  for (const serve of unused) {
    const eligible = permissive.map((interval, index) => ({ ...interval, id: `permissive:${index + 1}` })).filter(interval =>
      interval.end >= serve.time - config.associationSeconds && interval.start <= serve.time + config.associationSeconds
      && interval.end - interval.start <= config.maxRescueSeconds);
    if (eligible.length) {
      let selected = eligible[0];
      for (const candidate of eligible.slice(1)) {
        if (Math.min(Math.abs(candidate.start - serve.time), Math.abs(candidate.end - serve.time))
          < Math.min(Math.abs(selected.start - serve.time), Math.abs(selected.end - serve.time))) selected = candidate;
      }
      const origin = { kind: "serve-permissive-rescue", serve, permissiveInterval: selected,
        eligiblePermissiveIds: eligible.map(row => row.id), primaryAssociated: false };
      actions.push(origin);
      rows.push({ start: serve.time, end: Math.max(serve.time + sampleSeconds, selected.end),
        confidence: Math.max(serve.confidence, selected.confidence), origins: [origin] });
    } else if (config.fallbackSeconds > 0) {
      const origin = { kind: "serve-fallback-new-interval", serve, fallbackSeconds: config.fallbackSeconds,
        eligiblePermissiveIds: [], primaryAssociated: false };
      actions.push(origin);
      rows.push({ start: serve.time, end: serve.time + config.fallbackSeconds,
        confidence: serve.confidence, origins: [origin] });
    } else {
      actions.push({ kind: "unassociated-serve-not-created", serve });
    }
  }
  const ordered = rows.filter(row => row.end > row.start && row.start < duration && row.end > 0)
    .map(row => ({ ...row, start: Math.max(0, row.start), end: Math.min(duration, row.end) }))
    .sort((a, b) => a.start - b.start || a.end - b.end || a.confidence - b.confidence);
  const merged = [];
  for (const row of ordered) {
    const contribution = { intervalBeforeMerge: clean(row), origins: row.origins };
    const previous = merged.at(-1);
    if (previous && row.start < previous.end) {
      previous.end = Math.max(previous.end, row.end);
      previous.confidence = Math.max(previous.confidence, row.confidence);
      previous.contributions.push(contribution);
    } else merged.push({ ...clean(row), contributions: [contribution] });
  }
  return { intervals: merged, actions };
}

const components = {};
for (const [name, prefix, asset] of [
  ["previous", "PP-", "model-9c92b8e9333f.json"], ["v2", "AV2-", "model-1ca43e38eefc.json"],
]) {
  const bundle = model.loadOnDeviceModelBundle(JSON.parse(fs.readFileSync(new URL(input.repoUrl + "/prod/public/runtime/" + asset), "utf8")));
  const probabilities = Object.fromEntries(["rally", "serve", "deadState"].map(head => [head, Float32Array.from(input.probabilities[name + "_" + head])]));
  const primary = model.decodeProbabilities(times, probabilities.rally, input.duration, bundle.rally.decoder, fps);
  const permissive = model.decodeProbabilities(times, probabilities.rally, input.duration, bundle.serve.composition.permissiveDecoder, fps);
  const serves = model.decodeServeProbabilities(times, probabilities.serve, bundle.serve.decoder, input.duration);
  const composed = model.composeServeAnchoredIntervals(primary.intervals, permissive.intervals, serves, input.duration, bundle.serve.composition, sampleSeconds);
  const traced = traceComposition(primary.intervals, permissive.intervals, serves, input.duration, bundle.serve.composition);
  exact(traced.intervals.map(clean), composed.map(clean), `${name}: traced composition differs from production`);
  const refined = model.refineDeadStateEnds(times, probabilities.deadState, composed, input.duration, sampleSeconds, bundle.deadState.decoder, bundle.deadState.refinement);
  const noServe = model.refineDeadStateEnds(times, probabilities.deadState, primary.intervals, input.duration, sampleSeconds, bundle.deadState.decoder, bundle.deadState.refinement);
  const final = refined.intervals.map((interval, index) => ({ id: prefix + String(index + 1).padStart(3, "0"),
    ...clean(interval), included: true }));
  const expected = input.components[name === "v2" ? "allLabelsV2" : "previousProduction"];
  exact(final, expected, `${name}: final component replay differs`);
  exact(Array.from(primary.smoothed), input.probabilities[name + "_smoothedRally"], `${name}: smoothed scores differ`);
  const finalLineage = final.map((rally, index) => {
    const contributions = traced.intervals[index].contributions;
    const origins = contributions.flatMap(row => row.origins);
    const hasPrimary = origins.some(row => row.kind === "primary-rally");
    const rescueKinds = [...new Set(origins.filter(row => row.kind !== "primary-rally").map(row => row.kind))];
    return { ...rally, component: name, classification: hasPrimary ? "primary-rally-backed" : rescueKinds.join("+"),
      hasPrimaryRallyOrigin: hasPrimary, hasServeRescueOrigin: rescueKinds.length > 0,
      composedInterval: clean(composed[index]), endRefinementSeconds: rally.end - composed[index].end,
      deadStateTransition: refined.transitions[index] ?? null, contributions,
      primaryIntervalsOverlappingFinal: primary.intervals.map((row, i) => ({ id: `primary:${i + 1}`, ...row })).filter(row => overlap(row, rally)),
      noServeCounterfactualOverlaps: noServe.intervals.filter(row => overlap(row, rally)) };
  });
  components[name] = { runtimeAsset: asset, rallyDecoder: bundle.rally.decoder, serveDecoder: bundle.serve.decoder,
    serveComposition: bundle.serve.composition, deadStateRefinement: bundle.deadState.refinement,
    primaryIntervals: primary.intervals, permissiveIntervals: permissive.intervals, serves,
    serveActions: traced.actions, composedIntervals: composed, finalRallies: finalLineage,
    noServeCounterfactual: noServe, verification: { compositionExact: true, componentExact: true, smoothedProbabilitiesExact: true } };
}
const plain = rows => rows.map(({id, start, end, confidence, included}) => ({id, start, end, confidence, included}));
const union = ensemble.mergeProductionModelIntervals(plain(components.v2.finalRallies), plain(components.previous.finalRallies));
exact(union, input.unionWithConfidence, "Production union replay differs");
const noServeUnion = ensemble.mergeProductionModelIntervals(
  components.v2.noServeCounterfactual.intervals.map((row, index) => ({ id: `AV2-NOSERVE-${index + 1}`, ...row, included: true })),
  components.previous.noServeCounterfactual.intervals.map((row, index) => ({ id: `PP-NOSERVE-${index + 1}`, ...row, included: true })),
);
const all = [...components.v2.finalRallies, ...components.previous.finalRallies];
const lineage = union.map(rally => {
  const sources = all.filter(row => overlap(row, rally));
  const hasPrimary = sources.some(row => row.hasPrimaryRallyOrigin);
  return { ...rally, intersectsEvaluationTime: rally.end > input.ignoredEnd,
    retainedByAggressiveSuppression: input.aggressiveCore.some(row => row.start === rally.start && row.end === rally.end),
    mechanism: hasPrimary ? "rally-head-backed" : "serve-created-without-primary-rally",
    sourceKinds: [...new Set(sources.flatMap(row => row.contributions.flatMap(c => c.origins.map(origin => origin.kind))))],
    noServeCounterfactualOverlaps: noServeUnion.filter(row => overlap(row, rally)), sources };
});
process.stdout.write(JSON.stringify({ schemaVersion: 1, recordingId: input.id, durationSeconds: input.duration,
  replayMethod: "Replay checked-in production decoder, composition and dead-end refinement from saved native probabilities; independently trace every composition branch.",
  inferenceRerun: false, labelsRead: false, servingSideUsed: false, sampleSeconds, effectiveAnalysisFps: fps,
  ignoredIntervals: [{ start: 0, end: input.ignoredEnd }], components, unionLineage: lineage,
  noServeCounterfactualUnion: noServeUnion, verification: { componentEndpointsAndConfidenceExact: true, unionExact: true,
    tracedCompositionExact: true, probabilitySmoothingExact: true } }));
