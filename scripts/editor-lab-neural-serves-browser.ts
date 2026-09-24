import { openUrlMedia } from "../prod/src/lib/on-device/media.ts";
import { servingSideFramePlan, evaluateServingSideFrames } from "../prod/src/lib/on-device/serving-side.ts";
import { sampleSpecialistFramesSequentially } from "../prod/src/lib/on-device/specialist-frame-sampling.ts";

async function run() {
  const input = await (await fetch("/input.json")).json();
  const media = await openUrlMedia(input.videoUrl);
  const analyses = Object.fromEntries(Object.entries(input.populations).map(([key, intervals]) => [key, {
    intervals: intervals as any[], times: input.evidence.times, productionServeOutputs: input.evidence.serveOutputs,
  }]));
  const times = [...new Set(Object.values(analyses).flatMap(analysis => servingSideFramePlan(analysis, media.info.duration, null).requestedTimes))].sort((a, b) => a - b);
  console.log(`Sampling ${times.length} native source timestamps for all neural rally starts`);
  try {
    const sampled = await sampleSpecialistFramesSequentially(media, input.roi, { servingSideTimes: times }, (done, total) => {
      if (done % 100 === 0 || done === total) console.log(`Frames ${done}/${total}`);
    });
    const outputs: Record<string, unknown> = {};
    const counts: Record<string, unknown> = {};
    let modelId = "";
    for (const [key, analysis] of Object.entries(analyses)) {
      const { output } = await evaluateServingSideFrames(media.info.duration, analysis, null, sampled.servingSide,
        (done, total) => { if (done % 10 === 0 || done === total) console.log(`${key} features ${done}/${total}`); });
      modelId = output.modelId;
      outputs[key] = { ...output, features: { ...output.features, values: Array.from(output.features.values) } };
      counts[key] = Object.fromEntries(["near", "far", "review", "not-serve"].map(verdict => [verdict, output.candidates.filter(row => row.verdict === verdict).length]));
    }
    (window as any).neuralServeResult = { outputs, counts, modelId, recordingId: input.recordingId,
      manifestSha256: input.manifestSha256, serveEvidenceSha256: input.serveEvidenceSha256,
      labelsUsed: false, roi: input.roi, sourceSampling: "Production browser sequential decoder and native PTS",
      normalization: "Separate within-recording ranks for each complete neural rally population", createdAt: new Date().toISOString() };
  } finally { media.input.dispose(); }
}
run().catch(error => { console.error(error); (window as any).neuralServeError = String(error.stack ?? error); });
