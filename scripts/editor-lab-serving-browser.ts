import { openUrlMedia } from "../prod/src/lib/on-device/media.ts";
import { servingSideFramePlan, evaluateServingSideFrames, loadServingSideRuntime } from "../prod/src/lib/on-device/serving-side.ts";
import { composeServingSideVerdict, tiedPercentileRanks } from "../prod/src/lib/on-device/serving-side-model.ts";
import { sampleSpecialistFramesSequentially } from "../prod/src/lib/on-device/specialist-frame-sampling.ts";
import type { OnDeviceInterval } from "../prod/src/lib/on-device/types.ts";

(window as any).runLabServing = async (input: any) => {
  const media = await openUrlMedia(input.videoUrl);
  try {
    // Feature extraction is shared only for identical anchors. Ranking remains per population.
    const anchors = [...new Set<number>(Object.values(input.populations).flatMap((rows: any) => rows.map((r: any) => r.start)))].sort((a,b)=>a-b);
    const intervals = anchors.map((start, i) => ({ id: `anchor-${i}`, start, end: Math.min(input.duration, start + .01), included: true, confidence: 0 }));
    const analysis = { intervals, times: input.times, productionServeOutputs: input.serveOutputs };
    const plan = servingSideFramePlan(analysis, input.duration, null);
    console.log(`Serving side: ${anchors.length} unique anchors, ${plan.requestedTimes.length} frames`);
    const sampled = await sampleSpecialistFramesSequentially(media, input.roi, { servingSideTimes: plan.requestedTimes },
      (done, total) => { if (done % 500 === 0 || done === total) console.log(`Frames ${done}/${total}`); });
    // The shared sampler deduplicates at nanosecond precision. Different floating
    // additions can describe that same timestamp; restore its numeric aliases.
    const framesByKey = new Map([...sampled.servingSide].map(([time, frame]) => [time.toFixed(9), frame]));
    for (const time of plan.requestedTimes) {
      const frame = framesByKey.get(time.toFixed(9));
      if (!frame) throw new Error('Serving-side frame sampling is incomplete');
      sampled.servingSide.set(time, frame);
    }
    // From here onward only the small grayscale samples are needed.
    media.input.dispose();
    const { output } = await evaluateServingSideFrames(input.duration, analysis, null, sampled.servingSide,
      (done, total) => { if (done % 50 === 0 || done === total) console.log(`Features ${done}/${total}`); });
    const runtime = await loadServingSideRuntime();
    const columns = output.features.columns;
    const configurations: Record<string, unknown> = {};
    for (const [id, population] of Object.entries(input.populations) as [string, OnDeviceInterval[]][]) {
      const raw = new Float64Array(population.length * columns);
      population.forEach((row, i) => {
        const offset = anchors.indexOf(row.start) * columns;
        raw.set(output.features.values.subarray(offset, offset + columns), i * columns);
      });
      const ranked = tiedPercentileRanks(raw, population.length, columns);
      configurations[id] = { population, output: { modelId: runtime.modelId, modelFingerprint: runtime.fingerprint,
        candidates: population.map((row, i) => composeServingSideVerdict({ ...row, included: true, confidence: 0 },
          ranked.subarray(i * columns, (i + 1) * columns), input.times,
          input.serveOutputs.allLabelsV2, input.serveOutputs.previousProduction, runtime)) } };
    }
    return { schemaVersion: 1, recordingId: input.recordingId, contentSha256: input.contentSha256,
      duration: input.duration, labelsUsed: false, modelId: runtime.modelId, modelFingerprint: runtime.fingerprint,
      evidenceSha256: input.evidenceSha256, roi: input.roi, configurations,
      normalization: "Separate tied within-recording ranks for each complete model rally population",
      uniqueAnchors: anchors.length, sampledFrames: plan.requestedTimes.length };
  } finally { media.input.dispose(); }
};
