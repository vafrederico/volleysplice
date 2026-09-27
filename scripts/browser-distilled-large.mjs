// Research-only end-to-end browser runner. Model bytes and media are supplied by
// the isolated loopback harness; this module is not imported by the product app.
import { VideoSampleSink } from 'mediabunny';
import { openLocalMedia, openUrlMedia } from '../prod/src/lib/on-device/media.ts';
import { extractBrowserFeatures } from '../prod/src/lib/on-device/pipeline.ts';
import { contextualizeFeatures } from '../prod/src/lib/on-device/feature-math.ts';
import { runProductionInferenceFromFeatures } from '../prod/src/lib/on-device/production-inference.ts';
import { nearestSamplesAtTimestampsFromSequentialPass } from '../prod/src/lib/on-device/sequential-samples.ts';
import { loadOpenCv } from '../prod/src/lib/on-device/visual-features.ts';
import { geometry, normalizedLetterbox, regionalPoolWeights, roundTokensToFloat16,
  fuseFeatures, decodeRallies, INPUT_SIZE, TOKEN_DIMENSION, FUSED_DIMENSION } from './browser-distilled-large-contract.mjs';

function finite(values, name) {
  for (const value of values) if (!Number.isFinite(value)) throw new Error(`Nonfinite ${name}`);
  return values;
}

export function prepareImage(cv, imageData, roi = [0, 0, 1, 1], ptsOffset = 0) {
  const shape = geometry(imageData.width, imageData.height, roi);
  const source = cv.matFromImageData(imageData);
  const crop = source.roi(new cv.Rect(shape.x, shape.y, shape.cropWidth, shape.cropHeight));
  const rgb = new cv.Mat(), resized = new cv.Mat(), gray = new cv.Mat();
  const floatGray = new cv.Mat(), laplacian = new cv.Mat();
  try {
    cv.cvtColor(crop, rgb, cv.COLOR_RGBA2RGB);
    cv.resize(rgb, resized, new cv.Size(shape.resizedWidth, shape.resizedHeight), 0, 0, cv.INTER_LINEAR);
    const image = normalizedLetterbox(resized.data, shape);
    cv.cvtColor(resized, gray, cv.COLOR_RGB2GRAY);
    // Convert with the same float32 division as NumPy, not convertTo's double scale.
    floatGray.create(gray.rows, gray.cols, cv.CV_32FC1);
    const grayBytes = gray.data, grayFloats = floatGray.data32F;
    let sum = 0, clipped = 0;
    for (let i = 0; i < grayBytes.length; i++) {
      grayFloats[i] = grayBytes[i] / 255;
      sum += grayFloats[i];
      if (grayBytes[i] <= 2 || grayBytes[i] >= 253) clipped++;
    }
    const average = sum / grayBytes.length;
    let variance = 0;
    for (const value of grayFloats) variance += (value - average) ** 2;
    cv.Laplacian(floatGray, laplacian, cv.CV_32F, 1, 1, 0, cv.BORDER_DEFAULT);
    let lapSum = 0;
    for (const value of laplacian.data32F) lapSum += value;
    const lapMean = lapSum / laplacian.data32F.length;
    let lapVariance = 0;
    for (const value of laplacian.data32F) lapVariance += (value - lapMean) ** 2;
    const quality = new Float32Array([shape.resizedWidth * shape.resizedHeight / INPUT_SIZE ** 2,
      average, Math.sqrt(variance / grayBytes.length), lapVariance / laplacian.data32F.length,
      clipped / grayBytes.length, ptsOffset]);
    return { image, quality, poolWeights: regionalPoolWeights(shape.box), shape };
  } finally {
    laplacian.delete(); floatGray.delete(); gray.delete(); resized.delete(); rgb.delete(); crop.delete(); source.delete();
  }
}

async function download(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Asset unavailable: ${url}`);
  return response.arrayBuffer();
}

export async function initializeRuntime(options) {
  const ort = window.ort;
  if (ort.env.versions.web !== '1.22.0') throw new Error('Expected pinned ORT Web 1.22.0');
  if (crossOriginIsolated !== options.isolation) throw new Error('Actual browser isolation differs from the requested experiment');
  if (!crossOriginIsolated && options.wasmThreads !== 1) throw new Error('Nonisolated browser requires single-thread WASM');
  ort.env.wasm.numThreads = options.wasmThreads;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = `${location.origin}/ort/`;
  let adapterInfo = null;
  if (options.provider === 'webgpu') {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('Requested WebGPU adapter is unavailable; fallback is not allowed');
    if (adapter.isFallbackAdapter || adapter.info?.isFallbackAdapter) throw new Error('Software WebGPU adapter rejected');
    ort.env.webgpu.adapter = adapter;
    adapterInfo = adapter.info ? { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
      device: adapter.info.device, description: adapter.info.description, isFallbackAdapter: adapter.info.isFallbackAdapter } : {};
  } else if (options.provider !== 'wasm') throw new Error('Unknown provider');
  const sessionOptions = { executionProviders: options.provider === 'webgpu' && options.gpuLayout !== 'default'
      ? [{ name: 'webgpu', preferredLayout: options.gpuLayout }] : [options.provider],
    graphOptimizationLevel: options.graphOptimizationLevel };
  return { ort, sessionOptions, runtime: { version: ort.env.versions.web, requestedExecutionProviders: [options.provider],
    gpuPreferredLayout: options.gpuLayout, graphOptimizationLevel: options.graphOptimizationLevel,
    explicitFallbackProvider: false, wasmThreads: options.wasmThreads,
    requestedCrossOriginIsolation: options.isolation, crossOriginIsolated,
    adapter: adapterInfo, userAgent: navigator.userAgent,
    limitation: options.provider === 'webgpu'
      ? 'WebGPU is explicitly enabled with a real adapter. Operator placement is unverified; mixed execution with WASM/CPU fallback operators remains possible.'
      : 'WASM CPU execution; configured thread count and cross-origin isolation are recorded.' } };
}

export async function validateFixtures(options) {
  const { ort, runtime, sessionOptions } = await initializeRuntime(options), cv = await loadOpenCv();
  const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    value => value.toString(16).padStart(2, '0')).join('');
  const manifestBytes = await download('/fixtures/manifest.json');
  const fixtureManifestSha256 = await hash(manifestBytes);
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  if (manifest.schemaVersion !== 1 || manifest.kind !== 'distilled-large-browser-image-fixtures-v1'
      || !Array.isArray(manifest.cases) || !manifest.cases.length) throw new Error('Unexpected image fixture contract');
  const results = [];
  const verified = async entry => {
    if (!entry || !/^[a-z0-9-]+\.(f32|f64|u8)$/.test(entry.file)) throw new Error('Invalid fixture entry');
    const bytes = await download(`/fixtures/${entry.file}`);
    if (bytes.byteLength !== entry.sizeBytes || await hash(bytes) !== entry.sha256) throw new Error('Fixture identity changed');
    return bytes;
  };
  const array = async entry => new Float32Array(await verified(entry));
  const compare = (actual, expected, name, atol = 2e-6, rtol = 2e-6) => {
    if (actual.length !== expected.length) throw new Error(`${name}: shape differs`);
    let maxAbsoluteError = 0;
    for (let i = 0; i < actual.length; i++) {
      const error = Math.abs(actual[i] - expected[i]);
      if (!Number.isFinite(error) || error > atol + rtol * Math.abs(expected[i]))
        throw new Error(`${name}: parity failed at ${i}, actual ${actual[i]}, expected ${expected[i]}, error ${error}`);
      maxAbsoluteError = Math.max(maxAbsoluteError, error);
    }
    return { passed: true, maxAbsoluteError };
  };
  let session;
  try {
    if (options.validateEncoder) session = await ort.InferenceSession.create(
      await download('/graphs/mobile-large-encoder-fp32.onnx'), sessionOptions);
    for (const test of manifest.cases) {
      const rgba = new Uint8ClampedArray(await verified(test.rgba));
      const roi = Array.isArray(test.roi) ? test.roi : ['x', 'y', 'width', 'height'].map(k => test.roi[k]);
      const prepared = prepareImage(cv, new ImageData(rgba, test.width, test.height), roi, test.selectedPtsOffsetSeconds);
      const result = { name: test.id,
        pixels: compare(prepared.image, await array(test.expected.pixels), 'pixels'),
        quality: compare(prepared.quality, await array(test.expected.quality), 'quality'),
        poolWeights: compare(prepared.poolWeights, await array(test.expected.poolWeights), 'poolWeights'),
        contentBox: compare(prepared.shape.box, new Float64Array(await verified(test.expected.contentBox)), 'contentBox', 0, 0) };
      const expectedEncoder = test.encoderExpected?.[options.selection];
      if (session && !expectedEncoder) throw new Error('Requested encoder fixture is missing');
      if (session && expectedEncoder) {
        const image = new ort.Tensor('float32', prepared.image, [1, 3, 224, 224]);
        const pool = new ort.Tensor('float32', prepared.poolWeights, [1, 4, 7, 7]);
        const outputs = await session.run({ image, pool_weights: pool });
        try {
          const tokens = finite(new Float32Array(outputs.tokens.data), 'fixture tokens');
          const expectedTokens = await array(expectedEncoder.rawTokens);
          window.fixtureDiagnostics = { runtime, fixtureManifestSha256, caseId: test.id,
            tokenShape: outputs.tokens.dims, actualTokens: Array.from(tokens), expectedTokens: Array.from(expectedTokens) };
          result.rawTokens = compare(tokens, expectedTokens, 'rawTokens', 2e-4, 2e-4);
          result.roundedTokens = compare(roundTokensToFloat16(tokens), await array(expectedEncoder.roundedTokens), 'roundedTokens', .004, .002);
        } finally { image.dispose(); pool.dispose(); for (const value of Object.values(outputs)) value.dispose(); }
      }
      results.push(result);
    }
  } finally { await session?.release(); }
  return { schemaVersion: 1, passed: true, fixtureManifestSha256, runtime, cases: results };
}

/** Small operator controls isolate runtime semantics from model/preprocessing. */
export async function validateOperatorFixtures(options) {
  const { ort, runtime, sessionOptions } = await initializeRuntime(options);
  const manifestBytes = await download('/fixtures/manifest.json');
  const hash = async bytes => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)),
    value => value.toString(16).padStart(2, '0')).join('');
  const manifest = JSON.parse(new TextDecoder().decode(manifestBytes));
  if (manifest.schemaVersion !== 1 || manifest.kind !== 'distilled-large-webgpu-operator-probes-v1'
      || !Array.isArray(manifest.cases) || !manifest.cases.length) throw new Error('Unexpected operator fixture contract');
  const cases = [];
  for (const test of manifest.cases) {
    if (!/^[a-z0-9-]+\.onnx$/.test(test.graph.file)) throw new Error('Invalid operator graph filename');
    const graph = await download(`/fixtures/${test.graph.file}`);
    if (graph.byteLength !== test.graph.sizeBytes || await hash(graph) !== test.graph.sha256) throw new Error('Operator graph identity changed');
    const session = await ort.InferenceSession.create(graph, sessionOptions);
    const inputs = {};
    try {
      for (const input of test.inputs) {
        if (input.dtype !== 'float32') throw new Error('Operator probe expects float32');
        inputs[input.name] = new ort.Tensor('float32', new Float32Array(input.values), input.shape);
      }
      const outputs = await session.run(inputs);
      try {
        const output = outputs[test.expected.outputName], actual = Array.from(output.data);
        const shapeMatches = JSON.stringify(output.dims) === JSON.stringify(test.expected.shape);
        const errors = actual.map((value, i) => Math.abs(value - test.expected.values[i]));
        const passed = shapeMatches && actual.length === test.expected.values.length
          && errors.every((error, i) => Number.isFinite(error) && error <= 1e-6 + 1e-6 * Math.abs(test.expected.values[i]));
        cases.push({ id: test.id, passed, graphSha256: test.graph.sha256, actual, expected: test.expected.values,
          actualShape: output.dims, expectedShape: test.expected.shape, maxAbsoluteError: Math.max(...errors) });
      } finally { for (const output of Object.values(outputs)) output.dispose(); }
    } finally { for (const input of Object.values(inputs)) input.dispose(); await session.release(); }
  }
  return { schemaVersion: 1, passed: cases.every(test => test.passed), runtime,
    fixtureManifestSha256: await hash(manifestBytes), cases };
}

export async function runBrowserExperiment(options, scoreRunner) {
  const experimentStarted = performance.now();
  const { ort, runtime, sessionOptions } = await initializeRuntime(options);
  const runtimeInitializeMs = performance.now() - experimentStarted;
  const sourceProbeStarted = performance.now();
  const localFile = document.querySelector('#benchmark-source-file')?.files?.[0];
  if (options.sourceTransport === 'file' && !(localFile instanceof File)) throw new Error('Actual browser File input required');
  const media = options.sourceTransport === 'file' ? await openLocalMedia(localFile) : await openUrlMedia('/fixture.mp4');
  const sourceProbeMs = performance.now() - sourceProbeStarted;
  const analysisDurationSeconds = Math.min(options.seconds ?? media.info.duration, media.info.duration);
  if (!(analysisDurationSeconds > 0)) throw new Error('Invalid analysis window');
  const artifacts = {}, snapshots = [];
  let artifactIoMs = 0;
  const save = async (name, values) => {
    const start = performance.now();
    const bytes = values.buffer.slice(values.byteOffset, values.byteOffset + values.byteLength);
    const response = await fetch(`/artifacts/${name}`, { method: 'PUT', body: bytes });
    if (!response.ok) throw new Error(`Could not save ${name}`);
    artifacts[name] = await response.json(); artifactIoMs += performance.now() - start;
    return name;
  };
  const loadingStarted = performance.now();
  const config = await (await fetch('/graphs/mobile-large-pipeline.json')).json();
  if (config.modelIdentity !== 'dino-distilled-mobilenet-v3-large-tcn' || config.selectionMode !== options.selection
      || config.tokenDimension !== TOKEN_DIMENSION || config.recallTargetPercent !== 99)
    throw new Error('Frozen selected model identity differs');
  const graphFetchStarted = performance.now();
  let graphDownloadMs = 0, cvLoadMs = 0;
  const graphDownloads = Promise.all([
    download('/graphs/mobile-large-encoder-fp32.onnx'), download('/graphs/mobile-large-tcn-dynamic-fp32.onnx'),
  ]).then(values => { graphDownloadMs = performance.now() - graphFetchStarted; return values; });
  const cvLoading = loadOpenCv().then(value => { cvLoadMs = performance.now() - graphFetchStarted; return value; });
  const [encoderBytes, headBytes, cv] = await Promise.all([
    graphDownloads.then(values => values[0]), graphDownloads.then(values => values[1]), cvLoading]);
  const graphFetchAndCvLoadMs = performance.now() - graphFetchStarted;
  const encoderCreationStarted = performance.now();
  const encoder = await ort.InferenceSession.create(encoderBytes, sessionOptions);
  const encoderSessionCreationMs = performance.now() - encoderCreationStarted;
  let temporal, encoderReleased = false;
  const encoderNames = { inputNames: [...encoder.inputNames], outputNames: [...encoder.outputNames] };
  try {
    const headCreationStarted = performance.now();
    temporal = await ort.InferenceSession.create(headBytes, sessionOptions);
    const temporalSessionCreationMs = performance.now() - headCreationStarted;
    const modelLoadingMs = performance.now() - loadingStarted;
    const count = Math.ceil(analysisDurationSeconds * 2 - 1e-9);
    const times = Float64Array.from({ length: count }, (_, i) => i / 2);
    const pts = new Float64Array(count), tokensRaw = new Float32Array(count * TOKEN_DIMENSION);
    const quality = new Float32Array(count * 6);
    const canvas = new OffscreenCanvas(media.info.width, media.info.height);
    const context = canvas.getContext('2d', { alpha: false, willReadFrequently: true });
    if (!context) throw new Error('Encoder source canvas unavailable');
    const sink = new VideoSampleSink(media.videoTrack, { hardwareAcceleration: 'prefer-hardware' });
    let decodedFrames = 0, sampledFrames = 0, prepareMs = 0, canvasReadbackMs = 0, encoderAndReadbackMs = 0;
    let poolWeights;
    const snapshotIndexes = new Set([0, 1, Math.min(count - 1, 60), Math.floor(count / 2), count - 1]);
    const embeddingStarted = performance.now(), embeddingIoBefore = artifactIoMs;
    for await (const sample of nearestSamplesAtTimestampsFromSequentialPass(sink, Array.from(times), () => decodedFrames++, analysisDurationSeconds)) {
      if (!sample) throw new Error('Missing nearest encoder sample');
      try {
        if (Math.abs(sample.timestamp - times[sampledFrames]) > .25 + 1e-9)
          throw new Error('Encoder PTS exceeds half-tick tolerance');
        let start = performance.now();
        sample.draw(context, 0, 0, canvas.width, canvas.height);
        const rgba = context.getImageData(0, 0, canvas.width, canvas.height);
        canvasReadbackMs += performance.now() - start;
        start = performance.now();
        const prepared = prepareImage(cv, rgba, [0, 0, 1, 1], sample.timestamp - times[sampledFrames]);
        prepareMs += performance.now() - start;
        poolWeights ??= prepared.poolWeights;
        const image = new ort.Tensor('float32', prepared.image, [1, 3, 224, 224]);
        const pool = new ort.Tensor('float32', prepared.poolWeights, [1, 4, 7, 7]);
        start = performance.now();
        const outputs = await encoder.run({ image, pool_weights: pool });
        try {
          if (outputs.tokens.dims.join(',') !== '1,4,960') throw new Error('Encoder output dimensions differ');
          tokensRaw.set(finite(outputs.tokens.data, 'tokens'), sampledFrames * TOKEN_DIMENSION);
        } finally { image.dispose(); pool.dispose(); for (const value of Object.values(outputs)) value.dispose(); }
        encoderAndReadbackMs += performance.now() - start;
        quality.set(prepared.quality, sampledFrames * 6); pts[sampledFrames] = sample.timestamp;
        if (snapshotIndexes.has(sampledFrames)) {
          const prefix = `snapshot-${sampledFrames}`;
          snapshots.push({ sampleIndex: sampledFrames, pts: sample.timestamp, width: canvas.width, height: canvas.height,
            rgbaFile: await save(`${prefix}-rgba.u8`, rgba.data), imageFile: await save(`${prefix}-image.f32`, prepared.image),
            quality: Array.from(prepared.quality), contentBox: prepared.shape.box });
        }
        sampledFrames++;
        window.progress = { stage: 'embeddings', completed: sampledFrames, total: count };
      } finally { sample.close(); }
    }
    const embeddingWallMs = performance.now() - embeddingStarted, embeddingArtifactIoMs = artifactIoMs - embeddingIoBefore;
    if (sampledFrames !== count) throw new Error('Incomplete embedding grid');
    const roundingStarted = performance.now(), tokens = roundTokensToFloat16(tokensRaw);
    const tokenRoundingMs = performance.now() - roundingStarted;
    const embedding = { sampleCount: count, tokenDimension: TOKEN_DIMENSION,
      timesFile: await save('embedding-times.f64', times), selectedPtsFile: await save('embedding-pts.f64', pts),
      tokensFile: await save('embedding-tokens.f32', tokens), tokensUnroundedFile: await save('embedding-tokens-unrounded.f32', tokensRaw),
      qualityFile: await save('embedding-quality.f32', quality), poolWeightsFile: await save('embedding-pool-weights.f32', poolWeights),
      timings: { wallMs: embeddingWallMs, artifactIoMs: embeddingArtifactIoMs,
        pipelineMs: embeddingWallMs - embeddingArtifactIoMs, canvasReadbackMs, prepareMs, encoderAndReadbackMs,
        tokenRoundingMs, decodeAndOtherMs: embeddingWallMs - embeddingArtifactIoMs - canvasReadbackMs - prepareMs - encoderAndReadbackMs },
      decodedFrames, tokenStorage: 'FP32 graph compute; float16 roundtrip matches frozen training cache',
      sourcePixelContract: 'Browser native display-size RGB canvas; OpenCV INTER_LINEAR uint8 224 letterbox; nearest actual PTS' };
    await encoder.release(); encoderReleased = true;
    const avCases = [];
    for (const id of options.visualCases) {
      const visualPreprocessing = id === 'corrected' ? 'opencv-area-nearest-grid-v1' : 'canvas-preceding-v1';
      const avStarted = performance.now();
      const sequence = await extractBrowserFeatures(media, { x: 0, y: 0, width: 1, height: 1 }, undefined,
        p => { window.progress = { stage: `av-${id}-${p.stage}`, completed: p.completed, total: p.total }; }, undefined,
        { decodeStrategy: 'sequential', detailedProfiling: true, visualPreprocessing,
          analysisWindow: { start: 0, end: analysisDurationSeconds } });
      const avMs = performance.now() - avStarted;
      if (sequence.columns !== 104) throw new Error('AV104 schema differs');
      const fusionStarted = performance.now();
      const contextual = contextualizeFeatures(sequence.times, sequence.values, sequence.names);
      const ranked = new Float32Array(sequence.rows * 104);
      for (let row = 0; row < sequence.rows; row++) ranked.set(contextual.values.subarray(row * 520 + 208, row * 520 + 312), row * 104);
      const fused = fuseFeatures(sequence.times, ranked, times, tokens, quality, config);
      const fusionMs = performance.now() - fusionStarted;
      const headStarted = performance.now(), probabilities = new Float32Array(sequence.rows * 4);
      for (let core = 0; core < sequence.rows; core += 128) {
        const end = Math.min(sequence.rows, core + 128), left = Math.max(0, core - 62), right = Math.min(sequence.rows, end + 62);
        const input = new ort.Tensor('float32', fused.slice(left * FUSED_DIMENSION, right * FUSED_DIMENSION), [1, right - left, FUSED_DIMENSION]);
        const outputs = await temporal.run({ features: input });
        try {
          const logits = finite(outputs.logits.data, 'logits');
          for (let row = core; row < end; row++) for (let c = 0; c < 4; c++)
            probabilities[row * 4 + c] = 1 / (1 + Math.exp(-logits[(row - left) * 4 + c]));
        } finally { input.dispose(); for (const value of Object.values(outputs)) value.dispose(); }
      }
      const temporalMs = performance.now() - headStarted;
      const decoderStarted = performance.now(), rallies = decodeRallies(sequence.times, probabilities, analysisDurationSeconds, config.decoder);
      const decoderMs = performance.now() - decoderStarted;
      const productionStarted = performance.now();
      const production = await runProductionInferenceFromFeatures(sequence, { start: 0, end: analysisDurationSeconds });
      const productionMs = performance.now() - productionStarted;
      let scores = null;
      if (scoreRunner) scores = await scoreRunner(media, sequence, rallies, production, { start: 0, end: analysisDurationSeconds });
      avCases.push({ id, rows: sequence.rows, columns: sequence.columns, names: sequence.names,
        alignment: id === 'legacy'
          ? 'Legacy AV actual selected PTS; causal as-of nominal2Hz token alignment can lag at an early AV frame.'
          : 'Corrected AV nominal4Hz grid; causal hold on nominal2Hz tokens.',
        timesFile: await save(`${id}-times.f64`, sequence.times), rawAvFile: await save(`${id}-av-raw.f32`, sequence.values),
        rankedAvFile: await save(`${id}-av-ranked.f32`, ranked), fusedFile: await save(`${id}-fused.f32`, fused),
        probabilitiesFile: await save(`${id}-probabilities.f32`, probabilities), rallies, production, scores,
        timings: { avMs, fusionMs, temporalMs, decoderMs, productionMs, avPerformance: sequence.performance,
          neuralRalliesReadyMs: sourceProbeMs + runtimeInitializeMs + modelLoadingMs + embedding.timings.pipelineMs + tokenRoundingMs + avMs + fusionMs + temporalMs + decoderMs,
          ...(scores ? { neuralAllReadyMs: sourceProbeMs + runtimeInitializeMs + modelLoadingMs + embedding.timings.pipelineMs + tokenRoundingMs + avMs + fusionMs + temporalMs + decoderMs
              + productionMs + scores.neural.allReadyAdditionalMs,
            productionAllReadySharedPassMs: sourceProbeMs + avMs + productionMs + scores.production.allReadyAdditionalMs } : {}) } });
    }
    return { schemaVersion: 1, kind: 'distilled-large-real-video-browser-v1', model: config, media: media.info,
      analysisDurationSeconds, analysisWindow: { start: 0, end: analysisDurationSeconds }, runtime,
      sessions: { encoder: { ...encoderNames, inputShapes: [[1, 3, 224, 224], [1, 4, 7, 7]], outputShape: [1, 4, 960] },
        temporal: { inputNames: temporal.inputNames, outputNames: temporal.outputNames, inputShape: [1, 'ticks', 3952], outputShape: [1, 'ticks', 4] } },
      timings: { sourceProbeMs, runtimeInitializeMs, modelLoadingMs, graphDownloadMs, cvLoadMs,
        graphFetchAndCvLoadMs, encoderSessionCreationMs, temporalSessionCreationMs,
        wholeExperimentMs: performance.now() - experimentStarted, artifactIoMs },
      embedding, avCases, snapshots, artifacts, labelsUsed: false, trainingPerformed: false,
      limitations: ['Desktop Chrome experiment; no physical-phone browser timing.',
        'Raw browser RGB conversion can differ from OpenCV/Android YUV conversion; snapshots allow independent pixel and encoder replay.',
        'Research retains full feature and token arrays plus snapshots; this is not a mobile peak-memory implementation.',
        options.sourceTransport === 'file'
          ? 'Video-stage timings include direct File/BlobSource filesystem reads on configured storage; filesystem byte totals are unmeasured and this is not a decode-only microbenchmark.'
          : 'Video-stage timings include loopback reads of the original source on configured storage; they are not decode-only microbenchmarks.',
        'AV variants reuse the same encoder pass; ready times account each variant independently.',
        'Production comparison reuses AV work after neural runtime initialization; its shared-pass estimate is not a separately measured cold launch.',
        'No skipped warmup calls; first inference/shader compilation is included.'] };
  } finally {
    if (!encoderReleased) await encoder.release();
    await temporal?.release(); media.input.dispose();
  }
}
