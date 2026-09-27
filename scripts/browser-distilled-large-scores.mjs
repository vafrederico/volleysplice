// Research-only wrapper around the exact production score-specialist entry point.
// Each rally proposal set gets its own frame sampling and specialist inference.
import { inferScoreSpecialists } from '../prod/src/lib/on-device/score-specialists.ts';
import { loadServingSideRuntime } from '../prod/src/lib/on-device/serving-side.ts';
import { loadSideSwitchRuntime } from '../prod/src/lib/on-device/side-switch-model.ts';

export async function runScoreModels(media, sequence, rallies, production, analysisWindow) {
  if (!production.productionServeOutputs || !production.productionComponents || !production.productionStateOutputs)
    throw new Error('Score specialists require both production serve and state outputs.');
  if (analysisWindow.start !== 0 || analysisWindow.end > media.info.duration)
    throw new Error('Score benchmark expects a zero-based analysis window within the supplied media.');
  const scopedMedia = { ...media, info: { ...media.info, duration: analysisWindow.end } };
  const modelStart = performance.now();
  await Promise.all([loadServingSideRuntime(), loadSideSwitchRuntime()]);
  const modelLoadMs = performance.now() - modelStart;
  const run = async (intervals, label) => {
    const started = performance.now();
    const progress = [];
    const record = specialist => update => {
      const prior = progress.at(-1);
      if (!prior || prior.specialist !== specialist || prior.stage !== update.stage)
        progress.push({ specialist, stage: update.stage, elapsedMs: performance.now() - started,
          completed: update.completed, total: update.total });
      else Object.assign(prior, { completed: update.completed, total: update.total,
        lastElapsedMs: performance.now() - started });
      window.progress = { stage: `scores-${label}-${specialist}-${update.stage}`,
        completed: update.completed, total: update.total };
    };
    const output = await inferScoreSpecialists(scopedMedia, { x: 0, y: 0, width: 1, height: 1 }, {
      servingSide: {
        analysis: { intervals, times: sequence.times, productionServeOutputs: production.productionServeOutputs },
        onProgress: record('serving-side'),
      },
      sideSwitch: {
        analysis: { intervals, times: sequence.times, deadStateProbabilities: production.deadStateProbabilities,
          productionComponents: production.productionComponents, productionStateOutputs: production.productionStateOutputs },
        onProgress: record('side-switch'),
      },
    });
    const pipelineMs = performance.now() - started;
    if (!output.servingSide || !output.sideSwitch) throw new Error('Score specialist output is incomplete.');
    return { pipelineMs, allReadyAdditionalMs: pipelineMs + modelLoadMs,
      rallyCount: intervals.length, progress, output,
      timingIncludes: ['shared additional video decode and sampling', 'serving-side features and inference',
        'side-switch features and inference'],
    };
  };
  const intervals = rallies.map((rally, index) => ({ ...rally,
    id: `N${String(index + 1).padStart(3, '0')}`, included: true }));
  const neural = await run(intervals, 'neural');
  const baseline = await run(production.intervals, 'production');
  return { modelLoadMs, neural, production: baseline,
    contract: 'Production inferScoreSpecialists; full-frame ROI; existing score sampler unchanged.',
    limitation: 'Neural and production timings are independent proposals, run sequentially with shared model loading; browser/runtime caches may be warmer for the second pass.' };
}
