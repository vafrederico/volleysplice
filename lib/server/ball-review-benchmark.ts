import { constants as fsConstants } from "node:fs";
import { createHash } from "node:crypto";
import { lstat, open, realpath } from "node:fs/promises";
import path from "node:path";

import {
  benchmarkConfigurationIds,
  type BallReviewBenchmarkBundle,
  type BenchmarkConfiguration,
  type BenchmarkConfigurationId,
  type BenchmarkFrameAnnotation,
  type BenchmarkSimilarityMetrics,
  type BenchmarkUsage,
} from "../ball-review-benchmark.ts";

const defaultBenchmarkRoot =
  "/mnt/freenas/volleycut/ball-presence-v1/reports/ball-review-effort-screen12-v1";
const defaultReportSha256 =
  "818641939aa3f8e50d901933c69ad4229ea4146aefc1bce480740c93b3062638";
const benchmarkId = "ball-review-effort-screen12-v1" as const;
const frameWidth = 960;
const frameHeight = 540;
const frameCount = 12;

const configurationSpecs: Record<
  BenchmarkConfigurationId,
  Pick<BenchmarkConfiguration, "model" | "effort">
> = {
  "sol-low": { model: "gpt-5.6-sol", effort: "low" },
  "sol-medium": { model: "gpt-5.6-sol", effort: "medium" },
  "sol-high": { model: "gpt-5.6-sol", effort: "high" },
  "sol-xhigh": { model: "gpt-5.6-sol", effort: "xhigh" },
  "sol-max": { model: "gpt-5.6-sol", effort: "max" },
  "terra-xhigh": { model: "gpt-5.6-terra", effort: "xhigh" },
  "terra-max": { model: "gpt-5.6-terra", effort: "max" },
  "luna-xhigh": { model: "gpt-5.6-luna", effort: "xhigh" },
  "luna-max": { model: "gpt-5.6-luna", effort: "max" },
};

const annotationStates = [
  "localizable",
  "fully_occluded",
  "out_of_frame",
  "indeterminate",
] as const;
const objectRoles = ["primary-court", "other-court", "unknown"] as const;
const objectVisibilities = [
  "clear",
  "motion-blurred",
  "partially-occluded",
] as const;

type JsonRecord = Record<string, unknown>;

type BenchmarkRoot = {
  configured: string;
  real: string;
};

type ArtifactHashes = {
  blindPackSha256: string;
  schemaSha256: string;
  sealedReferenceSha256: string;
  runs: Record<
    BenchmarkConfigurationId,
    {
      resultSha256: string;
      receiptSha256: string;
      eventsSha256: string;
      stderrSha256: string;
    }
  >;
};

type ValidatedReport = {
  sample: BallReviewBenchmarkBundle["sample"];
  configurations: BallReviewBenchmarkBundle["configurations"];
  pairwise: BallReviewBenchmarkBundle["pairwise"];
  metricSemantics: BallReviewBenchmarkBundle["metricSemantics"];
  limitations: string[];
  artifactIntegrity: ArtifactHashes;
};

type TrustedReport = {
  root: BenchmarkRoot;
  sha256: string;
  report: ValidatedReport;
};

type BlindPackFrame = {
  attachmentIndex: number;
  frameId: string;
  image: string;
  sha256: string;
};

type BlindPack = {
  policy: string;
  width: number;
  height: number;
  frames: BlindPackFrame[];
};

export type BallReviewBenchmarkImage = {
  bytes: Uint8Array;
  sha256: string;
  filename: string;
};

export class BallReviewBenchmarkNotFoundError extends Error {
  constructor(message = "Benchmark image not found") {
    super(message);
    this.name = "BallReviewBenchmarkNotFoundError";
  }
}

export class BallReviewBenchmarkValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "BallReviewBenchmarkValidationError";
  }
}

export class BallReviewBenchmarkUnavailableError extends Error {
  constructor(message = "Benchmark artifacts are unavailable") {
    super(message);
    this.name = "BallReviewBenchmarkUnavailableError";
  }
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function expectRecord(
  value: unknown,
  expectedKeys: readonly string[],
  where: string,
): JsonRecord {
  if (!isRecord(value)) {
    throw new BallReviewBenchmarkValidationError(`${where} must be an object`);
  }
  const actual = Object.keys(value).sort();
  const expected = [...expectedKeys].sort();
  if (
    actual.length !== expected.length ||
    actual.some((key, index) => key !== expected[index])
  ) {
    throw new BallReviewBenchmarkValidationError(
      `${where} has an unexpected shape`,
    );
  }
  return value;
}

function expectString(value: unknown, where: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new BallReviewBenchmarkValidationError(
      `${where} must be a non-empty string`,
    );
  }
  return value;
}

function expectFinite(value: unknown, where: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new BallReviewBenchmarkValidationError(
      `${where} must be a finite number`,
    );
  }
  return value;
}

function expectPositive(value: unknown, where: string): number {
  const result = expectFinite(value, where);
  if (result <= 0) {
    throw new BallReviewBenchmarkValidationError(`${where} must be positive`);
  }
  return result;
}

function expectNonnegativeInteger(value: unknown, where: string): number {
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0
  ) {
    throw new BallReviewBenchmarkValidationError(
      `${where} must be a nonnegative safe integer`,
    );
  }
  return value;
}

function expectProbability(value: unknown, where: string): number {
  const result = expectFinite(value, where);
  if (result < 0 || result > 1) {
    throw new BallReviewBenchmarkValidationError(
      `${where} must be between zero and one`,
    );
  }
  return result;
}

function expectNullableProbability(
  value: unknown,
  where: string,
): number | null {
  return value === null ? null : expectProbability(value, where);
}

function expectNullableNonnegative(
  value: unknown,
  where: string,
): number | null {
  if (value === null) return null;
  const result = expectFinite(value, where);
  if (result < 0) {
    throw new BallReviewBenchmarkValidationError(
      `${where} must be nonnegative`,
    );
  }
  return result;
}

function expectSha256(value: unknown, where: string): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) {
    throw new BallReviewBenchmarkValidationError(
      `${where} must be a lowercase SHA-256 digest`,
    );
  }
  return value;
}

function expectExactString<T extends string>(
  value: unknown,
  expected: T,
  where: string,
): T {
  if (value !== expected) {
    throw new BallReviewBenchmarkValidationError(
      `${where} does not match the frozen benchmark identity`,
    );
  }
  return expected;
}

function hasExactStringSet(
  value: JsonRecord,
  expected: readonly string[],
): boolean {
  const actual = Object.keys(value).sort();
  const wanted = [...expected].sort();
  return (
    actual.length === wanted.length &&
    actual.every((key, index) => key === wanted[index])
  );
}

function isWithin(parent: string, candidate: string): boolean {
  const relative = path.relative(parent, candidate);
  return (
    relative === "" ||
    (!relative.startsWith(`..${path.sep}`) &&
      relative !== ".." &&
      !path.isAbsolute(relative))
  );
}

function sha256(bytes: Uint8Array): string {
  return createHash("sha256").update(bytes).digest("hex");
}

async function resolveBenchmarkRoot(): Promise<BenchmarkRoot> {
  const configured = path.resolve(
    /* turbopackIgnore: true */
    process.env.VOLLEYCUT_BALL_BENCHMARK_ROOT ?? defaultBenchmarkRoot,
  );
  try {
    const info = await lstat(configured);
    if (info.isSymbolicLink() || !info.isDirectory()) {
      throw new BallReviewBenchmarkValidationError(
        "Benchmark root must be a real directory",
      );
    }
    return { configured, real: await realpath(configured) };
  } catch (error) {
    if (error instanceof BallReviewBenchmarkValidationError) throw error;
    throw new BallReviewBenchmarkUnavailableError();
  }
}

async function readContainedFile(
  root: BenchmarkRoot,
  segments: readonly string[],
  where: string,
): Promise<Uint8Array> {
  if (
    segments.length === 0 ||
    segments.some(
      (segment) =>
        segment.length === 0 ||
        segment === "." ||
        segment === ".." ||
        path.isAbsolute(segment) ||
        segment.includes("/") ||
        segment.includes("\\"),
    )
  ) {
    throw new BallReviewBenchmarkValidationError(
      `${where} has an invalid artifact identity`,
    );
  }
  const candidate = path.resolve(root.configured, ...segments);
  const expectedReal = path.resolve(root.real, ...segments);
  if (!isWithin(root.configured, candidate) || !isWithin(root.real, expectedReal)) {
    throw new BallReviewBenchmarkValidationError(
      `${where} escapes the benchmark root`,
    );
  }

  try {
    const info = await lstat(candidate);
    if (info.isSymbolicLink() || !info.isFile()) {
      throw new BallReviewBenchmarkValidationError(
        `${where} must be a regular, non-symlink file`,
      );
    }
    const candidateReal = await realpath(candidate);
    if (candidateReal !== expectedReal || !isWithin(root.real, candidateReal)) {
      throw new BallReviewBenchmarkValidationError(
        `${where} failed containment validation`,
      );
    }
    const handle = await open(
      candidate,
      fsConstants.O_RDONLY | fsConstants.O_NOFOLLOW,
    );
    try {
      const openedInfo = await handle.stat();
      if (!openedInfo.isFile()) {
        throw new BallReviewBenchmarkValidationError(
          `${where} must remain a regular file`,
        );
      }
      return new Uint8Array(await handle.readFile());
    } finally {
      await handle.close();
    }
  } catch (error) {
    if (error instanceof BallReviewBenchmarkValidationError) throw error;
    throw new BallReviewBenchmarkUnavailableError(`${where} is unavailable`);
  }
}

function parseJson(bytes: Uint8Array, where: string): unknown {
  try {
    return JSON.parse(Buffer.from(bytes).toString("utf8"));
  } catch {
    throw new BallReviewBenchmarkValidationError(`${where} is not valid JSON`);
  }
}

async function readHashedArtifact(
  root: BenchmarkRoot,
  segments: readonly string[],
  expectedSha256: string,
  where: string,
): Promise<Uint8Array> {
  const bytes = await readContainedFile(root, segments, where);
  if (sha256(bytes) !== expectedSha256) {
    throw new BallReviewBenchmarkValidationError(
      `${where} does not match its trusted SHA-256`,
    );
  }
  return bytes;
}

function validateUsage(value: unknown, where: string): BenchmarkUsage {
  const row = expectRecord(
    value,
    [
      "cache_write_input_tokens",
      "cached_input_tokens",
      "input_tokens",
      "output_tokens",
      "reasoning_output_tokens",
    ],
    where,
  );
  return {
    cache_write_input_tokens: expectNonnegativeInteger(
      row.cache_write_input_tokens,
      `${where}.cache_write_input_tokens`,
    ),
    cached_input_tokens: expectNonnegativeInteger(
      row.cached_input_tokens,
      `${where}.cached_input_tokens`,
    ),
    input_tokens: expectNonnegativeInteger(
      row.input_tokens,
      `${where}.input_tokens`,
    ),
    output_tokens: expectNonnegativeInteger(
      row.output_tokens,
      `${where}.output_tokens`,
    ),
    reasoning_output_tokens: expectNonnegativeInteger(
      row.reasoning_output_tokens,
      `${where}.reasoning_output_tokens`,
    ),
  };
}

function validateMetrics(
  value: unknown,
  where: string,
): BenchmarkSimilarityMetrics {
  const row = expectRecord(
    value,
    [
      "frameCount",
      "stateAccuracy",
      "stateMacroF1",
      "primaryPresenceF1",
      "boxF1Iou25",
      "boxF1Iou50",
      "matchedPrimaryCount",
      "matchedPrimaryMeanIou",
      "matchedPrimaryMedianCenterErrorPixels",
      "matchedPrimaryVisibilityAccuracyIou25",
      "objectCountAccuracy",
    ],
    where,
  );
  const validatedFrameCount = expectNonnegativeInteger(
    row.frameCount,
    `${where}.frameCount`,
  );
  if (validatedFrameCount !== frameCount) {
    throw new BallReviewBenchmarkValidationError(
      `${where}.frameCount does not match the benchmark sample`,
    );
  }
  const matchedPrimaryCount = expectNonnegativeInteger(
    row.matchedPrimaryCount,
    `${where}.matchedPrimaryCount`,
  );
  if (matchedPrimaryCount > frameCount) {
    throw new BallReviewBenchmarkValidationError(
      `${where}.matchedPrimaryCount exceeds the benchmark sample`,
    );
  }
  return {
    frameCount: validatedFrameCount,
    stateAccuracy: expectProbability(
      row.stateAccuracy,
      `${where}.stateAccuracy`,
    ),
    stateMacroF1: expectProbability(
      row.stateMacroF1,
      `${where}.stateMacroF1`,
    ),
    primaryPresenceF1: expectProbability(
      row.primaryPresenceF1,
      `${where}.primaryPresenceF1`,
    ),
    boxF1Iou25: expectProbability(
      row.boxF1Iou25,
      `${where}.boxF1Iou25`,
    ),
    boxF1Iou50: expectProbability(
      row.boxF1Iou50,
      `${where}.boxF1Iou50`,
    ),
    matchedPrimaryCount,
    matchedPrimaryMeanIou: expectNullableProbability(
      row.matchedPrimaryMeanIou,
      `${where}.matchedPrimaryMeanIou`,
    ),
    matchedPrimaryMedianCenterErrorPixels: expectNullableNonnegative(
      row.matchedPrimaryMedianCenterErrorPixels,
      `${where}.matchedPrimaryMedianCenterErrorPixels`,
    ),
    matchedPrimaryVisibilityAccuracyIou25: expectNullableProbability(
      row.matchedPrimaryVisibilityAccuracyIou25,
      `${where}.matchedPrimaryVisibilityAccuracyIou25`,
    ),
    objectCountAccuracy: expectProbability(
      row.objectCountAccuracy,
      `${where}.objectCountAccuracy`,
    ),
  };
}

function metricsEqual(
  first: BenchmarkSimilarityMetrics,
  second: BenchmarkSimilarityMetrics,
): boolean {
  return (Object.keys(first) as Array<keyof BenchmarkSimilarityMetrics>).every(
    (key) => Object.is(first[key], second[key]),
  );
}

function usageEqual(first: BenchmarkUsage, second: BenchmarkUsage): boolean {
  return (Object.keys(first) as Array<keyof BenchmarkUsage>).every(
    (key) => first[key] === second[key],
  );
}

function validateArtifactIntegrity(value: unknown): ArtifactHashes {
  const integrity = expectRecord(
    value,
    [
      "blindPackSha256",
      "schemaSha256",
      "sealedReferenceSha256",
      "runs",
    ],
    "report.artifactIntegrity",
  );
  const rawRuns = expectRecord(
    integrity.runs,
    benchmarkConfigurationIds,
    "report.artifactIntegrity.runs",
  );
  const runs = {} as ArtifactHashes["runs"];
  for (const id of benchmarkConfigurationIds) {
    const row = expectRecord(
      rawRuns[id],
      ["resultSha256", "receiptSha256", "eventsSha256", "stderrSha256"],
      `report.artifactIntegrity.runs.${id}`,
    );
    runs[id] = {
      resultSha256: expectSha256(
        row.resultSha256,
        `report.artifactIntegrity.runs.${id}.resultSha256`,
      ),
      receiptSha256: expectSha256(
        row.receiptSha256,
        `report.artifactIntegrity.runs.${id}.receiptSha256`,
      ),
      eventsSha256: expectSha256(
        row.eventsSha256,
        `report.artifactIntegrity.runs.${id}.eventsSha256`,
      ),
      stderrSha256: expectSha256(
        row.stderrSha256,
        `report.artifactIntegrity.runs.${id}.stderrSha256`,
      ),
    };
  }
  return {
    blindPackSha256: expectSha256(
      integrity.blindPackSha256,
      "report.artifactIntegrity.blindPackSha256",
    ),
    schemaSha256: expectSha256(
      integrity.schemaSha256,
      "report.artifactIntegrity.schemaSha256",
    ),
    sealedReferenceSha256: expectSha256(
      integrity.sealedReferenceSha256,
      "report.artifactIntegrity.sealedReferenceSha256",
    ),
    runs,
  };
}

function validateReport(value: unknown): ValidatedReport {
  const report = expectRecord(
    value,
    [
      "schemaVersion",
      "status",
      "sample",
      "configurations",
      "pairwise",
      "metricSemantics",
      "artifactIntegrity",
      "limitations",
    ],
    "report",
  );
  if (report.schemaVersion !== 1 || report.status !== "complete") {
    throw new BallReviewBenchmarkValidationError(
      "Benchmark report is not a completed schema-v1 report",
    );
  }

  const artifactIntegrity = validateArtifactIntegrity(report.artifactIntegrity);
  const rawSample = expectRecord(
    report.sample,
    [
      "frameCount",
      "recordingCount",
      "environments",
      "reference",
      "sealedReferenceSha256",
    ],
    "report.sample",
  );
  if (rawSample.frameCount !== frameCount || rawSample.recordingCount !== 8) {
    throw new BallReviewBenchmarkValidationError(
      "Benchmark sample cardinality does not match the frozen screen",
    );
  }
  if (
    !Array.isArray(rawSample.environments) ||
    rawSample.environments.length !== 3 ||
    rawSample.environments.some((row) => typeof row !== "string") ||
    rawSample.environments.join("\u0000") !== "beach\u0000grass\u0000indoor"
  ) {
    throw new BallReviewBenchmarkValidationError(
      "Benchmark environments do not match the frozen screen",
    );
  }
  const sampleReference = expectExactString(
    rawSample.reference,
    "prior blind Sol-xhigh pseudo-reference, not human truth",
    "report.sample.reference",
  );
  const sampleReferenceSha = expectSha256(
    rawSample.sealedReferenceSha256,
    "report.sample.sealedReferenceSha256",
  );
  if (sampleReferenceSha !== artifactIntegrity.sealedReferenceSha256) {
    throw new BallReviewBenchmarkValidationError(
      "Report reference hashes disagree",
    );
  }
  const sample: BallReviewBenchmarkBundle["sample"] = {
    frameCount,
    recordingCount: 8,
    environments: ["beach", "grass", "indoor"],
    reference: sampleReference,
    sealedReferenceSha256: sampleReferenceSha,
  };

  const rawConfigurations = expectRecord(
    report.configurations,
    benchmarkConfigurationIds,
    "report.configurations",
  );
  const configurations = {} as BallReviewBenchmarkBundle["configurations"];
  for (const id of benchmarkConfigurationIds) {
    const where = `report.configurations.${id}`;
    const raw = expectRecord(
      rawConfigurations[id],
      [
        "model",
        "effort",
        "elapsedSeconds",
        "speedupVsFreshSolXhigh",
        "usage",
        "similarityToPriorSolXhigh",
      ],
      where,
    );
    const spec = configurationSpecs[id];
    expectExactString(raw.model, spec.model, `${where}.model`);
    expectExactString(raw.effort, spec.effort, `${where}.effort`);
    configurations[id] = {
      ...spec,
      elapsedSeconds: expectPositive(
        raw.elapsedSeconds,
        `${where}.elapsedSeconds`,
      ),
      speedupVsFreshSolXhigh: expectPositive(
        raw.speedupVsFreshSolXhigh,
        `${where}.speedupVsFreshSolXhigh`,
      ),
      usage: validateUsage(raw.usage, `${where}.usage`),
      similarityToPriorSolXhigh: validateMetrics(
        raw.similarityToPriorSolXhigh,
        `${where}.similarityToPriorSolXhigh`,
      ),
    };
  }
  const baseline = configurations["sol-xhigh"].elapsedSeconds;
  for (const id of benchmarkConfigurationIds) {
    const expectedSpeedup = baseline / configurations[id].elapsedSeconds;
    if (
      Math.abs(
        expectedSpeedup - configurations[id].speedupVsFreshSolXhigh,
      ) >
      Math.max(1, expectedSpeedup) * 1e-12
    ) {
      throw new BallReviewBenchmarkValidationError(
        `report.configurations.${id} has an inconsistent speedup`,
      );
    }
  }

  const rawPairwise = expectRecord(
    report.pairwise,
    benchmarkConfigurationIds,
    "report.pairwise",
  );
  const pairwise = {} as BallReviewBenchmarkBundle["pairwise"];
  for (const first of benchmarkConfigurationIds) {
    const rawRow = expectRecord(
      rawPairwise[first],
      benchmarkConfigurationIds,
      `report.pairwise.${first}`,
    );
    const row = {} as Record<
      BenchmarkConfigurationId,
      BenchmarkSimilarityMetrics
    >;
    for (const second of benchmarkConfigurationIds) {
      row[second] = validateMetrics(
        rawRow[second],
        `report.pairwise.${first}.${second}`,
      );
    }
    pairwise[first] = row;
  }
  for (const first of benchmarkConfigurationIds) {
    for (const second of benchmarkConfigurationIds) {
      if (!metricsEqual(pairwise[first][second], pairwise[second][first])) {
        throw new BallReviewBenchmarkValidationError(
          `report pairwise matrix is not symmetric at ${first}/${second}`,
        );
      }
    }
  }

  const semantics = expectRecord(
    report.metricSemantics,
    ["similarityToPriorSolXhigh", "pairwise"],
    "report.metricSemantics",
  );
  const metricSemantics = {
    similarityToPriorSolXhigh: expectString(
      semantics.similarityToPriorSolXhigh,
      "report.metricSemantics.similarityToPriorSolXhigh",
    ),
    pairwise: expectString(
      semantics.pairwise,
      "report.metricSemantics.pairwise",
    ),
  };

  if (
    !Array.isArray(report.limitations) ||
    report.limitations.length === 0 ||
    report.limitations.some(
      (limitation) => typeof limitation !== "string" || limitation.length === 0,
    )
  ) {
    throw new BallReviewBenchmarkValidationError(
      "report.limitations must contain non-empty strings",
    );
  }

  return {
    sample,
    configurations,
    pairwise,
    metricSemantics,
    limitations: [...report.limitations] as string[],
    artifactIntegrity,
  };
}

async function loadTrustedReport(): Promise<TrustedReport> {
  const root = await resolveBenchmarkRoot();
  const expectedSha = expectSha256(
    process.env.VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256 ?? defaultReportSha256,
    "VOLLEYCUT_BALL_BENCHMARK_REPORT_SHA256",
  );
  const bytes = await readContainedFile(root, ["report.json"], "report.json");
  const actualSha = sha256(bytes);
  if (actualSha !== expectedSha) {
    throw new BallReviewBenchmarkValidationError(
      "report.json does not match the compiled trusted SHA-256",
    );
  }
  return {
    root,
    sha256: actualSha,
    report: validateReport(parseJson(bytes, "report.json")),
  };
}

function validateBlindPack(value: unknown): BlindPack {
  const pack = expectRecord(
    value,
    ["schemaVersion", "policy", "width", "height", "frames"],
    "blind-pack.json",
  );
  if (
    pack.schemaVersion !== 1 ||
    pack.policy !== "volleyball-ball-presence-v1" ||
    pack.width !== frameWidth ||
    pack.height !== frameHeight
  ) {
    throw new BallReviewBenchmarkValidationError(
      "blind-pack.json identity does not match the frozen benchmark",
    );
  }
  if (!Array.isArray(pack.frames) || pack.frames.length !== frameCount) {
    throw new BallReviewBenchmarkValidationError(
      "blind-pack.json must contain exactly twelve frames",
    );
  }
  const frameIds = new Set<string>();
  const frames = pack.frames.map((candidate, zeroBasedIndex) => {
    const where = `blind-pack.json.frames[${zeroBasedIndex}]`;
    const row = expectRecord(
      candidate,
      ["attachmentIndex", "frameId", "image", "sha256"],
      where,
    );
    const attachmentIndex = zeroBasedIndex + 1;
    if (row.attachmentIndex !== attachmentIndex) {
      throw new BallReviewBenchmarkValidationError(
        `${where}.attachmentIndex is out of order`,
      );
    }
    const frameId = expectString(row.frameId, `${where}.frameId`);
    if (!/^f[0-9]{9}$/.test(frameId) || frameIds.has(frameId)) {
      throw new BallReviewBenchmarkValidationError(
        `${where}.frameId is invalid or duplicated`,
      );
    }
    frameIds.add(frameId);
    const image = `image-${String(attachmentIndex).padStart(2, "0")}.png`;
    expectExactString(row.image, image, `${where}.image`);
    return {
      attachmentIndex,
      frameId,
      image,
      sha256: expectSha256(row.sha256, `${where}.sha256`),
    };
  });
  return {
    policy: "volleyball-ball-presence-v1",
    width: frameWidth,
    height: frameHeight,
    frames,
  };
}

async function loadBlindPack(trusted: TrustedReport): Promise<BlindPack> {
  const bytes = await readHashedArtifact(
    trusted.root,
    ["blind-pack.json"],
    trusted.report.artifactIntegrity.blindPackSha256,
    "blind-pack.json",
  );
  return validateBlindPack(parseJson(bytes, "blind-pack.json"));
}

function validateAnnotation(
  value: unknown,
  where: string,
): BenchmarkFrameAnnotation {
  const row = expectRecord(
    value,
    ["status", "primaryBallState", "objects", "notes"],
    where,
  );
  expectExactString(row.status, "reviewed", `${where}.status`);
  if (
    typeof row.primaryBallState !== "string" ||
    !(annotationStates as readonly string[]).includes(row.primaryBallState)
  ) {
    throw new BallReviewBenchmarkValidationError(
      `${where}.primaryBallState is invalid`,
    );
  }
  if (!Array.isArray(row.objects) || row.objects.length > 8) {
    throw new BallReviewBenchmarkValidationError(
      `${where}.objects must be an array of at most eight objects`,
    );
  }
  const objectIds = new Set<string>();
  const objects = row.objects.map((candidate, index) => {
    const objectWhere = `${where}.objects[${index}]`;
    const object = expectRecord(
      candidate,
      ["id", "category", "role", "bbox", "visibility", "truncated"],
      objectWhere,
    );
    const id = expectString(object.id, `${objectWhere}.id`);
    if (objectIds.has(id)) {
      throw new BallReviewBenchmarkValidationError(
        `${objectWhere}.id is duplicated within the frame`,
      );
    }
    objectIds.add(id);
    expectExactString(
      object.category,
      "volleyball",
      `${objectWhere}.category`,
    );
    if (
      typeof object.role !== "string" ||
      !(objectRoles as readonly string[]).includes(object.role)
    ) {
      throw new BallReviewBenchmarkValidationError(
        `${objectWhere}.role is invalid`,
      );
    }
    if (
      typeof object.visibility !== "string" ||
      !(objectVisibilities as readonly string[]).includes(object.visibility)
    ) {
      throw new BallReviewBenchmarkValidationError(
        `${objectWhere}.visibility is invalid`,
      );
    }
    if (typeof object.truncated !== "boolean") {
      throw new BallReviewBenchmarkValidationError(
        `${objectWhere}.truncated must be boolean`,
      );
    }
    const rawBox = expectRecord(
      object.bbox,
      ["x", "y", "width", "height"],
      `${objectWhere}.bbox`,
    );
    const x = expectFinite(rawBox.x, `${objectWhere}.bbox.x`);
    const y = expectFinite(rawBox.y, `${objectWhere}.bbox.y`);
    const width = expectPositive(rawBox.width, `${objectWhere}.bbox.width`);
    const height = expectPositive(rawBox.height, `${objectWhere}.bbox.height`);
    if (
      x < 0 ||
      y < 0 ||
      x > 1 ||
      y > 1 ||
      width > 1 ||
      height > 1 ||
      x + width > 1 + 1e-9 ||
      y + height > 1 + 1e-9
    ) {
      throw new BallReviewBenchmarkValidationError(
        `${objectWhere}.bbox must fit inside the normalized image`,
      );
    }
    return {
      id,
      category: "volleyball" as const,
      role: object.role as (typeof objectRoles)[number],
      bbox: { x, y, width, height },
      visibility: object.visibility as (typeof objectVisibilities)[number],
      truncated: object.truncated,
    };
  });
  const primaryCount = objects.filter(
    (object) => object.role === "primary-court",
  ).length;
  const expectedPrimaryCount =
    row.primaryBallState === "localizable" ? 1 : 0;
  if (primaryCount !== expectedPrimaryCount) {
    throw new BallReviewBenchmarkValidationError(
      `${where} has an inconsistent primary state/object count`,
    );
  }
  if (typeof row.notes !== "string" || row.notes.length > 10_000) {
    throw new BallReviewBenchmarkValidationError(
      `${where}.notes must be a bounded string`,
    );
  }
  return {
    status: "reviewed",
    primaryBallState:
      row.primaryBallState as BenchmarkFrameAnnotation["primaryBallState"],
    objects,
    notes: row.notes,
  };
}

function validateReference(
  value: unknown,
  expectedFrameIds: readonly string[],
): Record<string, BenchmarkFrameAnnotation> {
  const reference = expectRecord(
    value,
    ["schemaVersion", "source", "sourceTaskSha256", "frames"],
    "sealed-reference.json",
  );
  if (
    reference.schemaVersion !== 1 ||
    reference.source !== "prior blind Sol xhigh pseudo-reference"
  ) {
    throw new BallReviewBenchmarkValidationError(
      "sealed-reference.json identity is invalid",
    );
  }
  if (!isRecord(reference.sourceTaskSha256)) {
    throw new BallReviewBenchmarkValidationError(
      "sealed-reference.json source identities are invalid",
    );
  }
  const sourceEntries = Object.entries(reference.sourceTaskSha256);
  if (
    sourceEntries.length !== 8 ||
    sourceEntries.some(
      ([sourceId, digest]) =>
        sourceId.length === 0 ||
        typeof digest !== "string" ||
        !/^[a-f0-9]{64}$/.test(digest),
    )
  ) {
    throw new BallReviewBenchmarkValidationError(
      "sealed-reference.json source identities are invalid",
    );
  }
  if (
    !isRecord(reference.frames) ||
    !hasExactStringSet(reference.frames, expectedFrameIds)
  ) {
    throw new BallReviewBenchmarkValidationError(
      "sealed-reference.json frame coverage is invalid",
    );
  }
  const frames: Record<string, BenchmarkFrameAnnotation> = {};
  for (const frameId of expectedFrameIds) {
    frames[frameId] = validateAnnotation(
      reference.frames[frameId],
      `sealed-reference.json.frames.${frameId}`,
    );
  }
  return frames;
}

function validateResult(
  value: unknown,
  expectedFrameIds: readonly string[],
  configId: BenchmarkConfigurationId,
): Record<string, BenchmarkFrameAnnotation> {
  const result = expectRecord(
    value,
    ["schemaVersion", "frames"],
    `${configId}/result.json`,
  );
  if (result.schemaVersion !== 1 || !Array.isArray(result.frames)) {
    throw new BallReviewBenchmarkValidationError(
      `${configId}/result.json identity is invalid`,
    );
  }
  if (result.frames.length !== expectedFrameIds.length) {
    throw new BallReviewBenchmarkValidationError(
      `${configId}/result.json frame count is invalid`,
    );
  }
  const frames: Record<string, BenchmarkFrameAnnotation> = {};
  result.frames.forEach((candidate, index) => {
    const where = `${configId}/result.json.frames[${index}]`;
    if (!isRecord(candidate)) {
      throw new BallReviewBenchmarkValidationError(`${where} must be an object`);
    }
    const { frameId: rawFrameId, ...annotation } = candidate;
    const expectedFrameId = expectedFrameIds[index];
    expectExactString(rawFrameId, expectedFrameId, `${where}.frameId`);
    frames[expectedFrameId] = validateAnnotation(annotation, where);
  });
  return frames;
}

function validateReceipt(
  value: unknown,
  configId: BenchmarkConfigurationId,
  resultSha256: string,
  expectedConfiguration: BenchmarkConfiguration,
): void {
  const where = `${configId}/receipt.json`;
  const receipt = expectRecord(
    value,
    [
      "schemaVersion",
      "configId",
      "model",
      "effort",
      "serviceTier",
      "startedEpochSeconds",
      "elapsedSeconds",
      "exitCode",
      "valid",
      "usage",
      "resultSha256",
    ],
    where,
  );
  const spec = configurationSpecs[configId];
  if (
    receipt.schemaVersion !== 1 ||
    receipt.configId !== configId ||
    receipt.model !== spec.model ||
    receipt.effort !== spec.effort ||
    receipt.serviceTier !== "default" ||
    receipt.exitCode !== 0 ||
    receipt.valid !== true ||
    receipt.resultSha256 !== resultSha256
  ) {
    throw new BallReviewBenchmarkValidationError(
      `${where} identity or result provenance is invalid`,
    );
  }
  expectPositive(receipt.startedEpochSeconds, `${where}.startedEpochSeconds`);
  const elapsed = expectPositive(receipt.elapsedSeconds, `${where}.elapsedSeconds`);
  if (elapsed !== expectedConfiguration.elapsedSeconds) {
    throw new BallReviewBenchmarkValidationError(
      `${where} elapsed time differs from report.json`,
    );
  }
  const usage = validateUsage(receipt.usage, `${where}.usage`);
  if (!usageEqual(usage, expectedConfiguration.usage)) {
    throw new BallReviewBenchmarkValidationError(
      `${where} usage differs from report.json`,
    );
  }
}

function validateResultSchema(value: unknown): void {
  const schema = expectRecord(
    value,
    ["$schema", "type", "additionalProperties", "required", "properties"],
    "result.schema.json",
  );
  if (
    schema.$schema !== "https://json-schema.org/draft/2020-12/schema" ||
    schema.type !== "object" ||
    schema.additionalProperties !== false ||
    !Array.isArray(schema.required) ||
    schema.required.join("\u0000") !== "schemaVersion\u0000frames" ||
    !isRecord(schema.properties)
  ) {
    throw new BallReviewBenchmarkValidationError(
      "result.schema.json does not describe the benchmark result envelope",
    );
  }
}

function validatePng(bytes: Uint8Array, where: string): void {
  const signature = [137, 80, 78, 71, 13, 10, 26, 10];
  if (
    bytes.byteLength < 24 ||
    signature.some((value, index) => bytes[index] !== value) ||
    bytes[8] !== 0 ||
    bytes[9] !== 0 ||
    bytes[10] !== 0 ||
    bytes[11] !== 13 ||
    String.fromCharCode(bytes[12], bytes[13], bytes[14], bytes[15]) !== "IHDR"
  ) {
    throw new BallReviewBenchmarkValidationError(
      `${where} is not a PNG with an IHDR header`,
    );
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (view.getUint32(16) !== frameWidth || view.getUint32(20) !== frameHeight) {
    throw new BallReviewBenchmarkValidationError(
      `${where} dimensions do not match 960x540`,
    );
  }
}

async function readAndValidateImage(
  root: BenchmarkRoot,
  frame: BlindPackFrame,
): Promise<BallReviewBenchmarkImage> {
  const where = `inputs/${frame.image}`;
  const bytes = await readHashedArtifact(
    root,
    ["inputs", frame.image],
    frame.sha256,
    where,
  );
  validatePng(bytes, where);
  return { bytes, sha256: frame.sha256, filename: frame.image };
}

export async function getBallReviewBenchmarkBundle(): Promise<BallReviewBenchmarkBundle> {
  const trusted = await loadTrustedReport();
  const pack = await loadBlindPack(trusted);
  const expectedFrameIds = pack.frames.map((frame) => frame.frameId);

  const schemaBytes = await readHashedArtifact(
    trusted.root,
    ["result.schema.json"],
    trusted.report.artifactIntegrity.schemaSha256,
    "result.schema.json",
  );
  validateResultSchema(parseJson(schemaBytes, "result.schema.json"));

  const referenceBytes = await readHashedArtifact(
    trusted.root,
    ["sealed-reference.json"],
    trusted.report.artifactIntegrity.sealedReferenceSha256,
    "sealed-reference.json",
  );
  const reference = validateReference(
    parseJson(referenceBytes, "sealed-reference.json"),
    expectedFrameIds,
  );

  const outputs = {} as Record<
    BenchmarkConfigurationId,
    Record<string, BenchmarkFrameAnnotation>
  >;
  await Promise.all(
    benchmarkConfigurationIds.map(async (configId) => {
      const hashes = trusted.report.artifactIntegrity.runs[configId];
      const [resultBytes, receiptBytes] = await Promise.all([
        readHashedArtifact(
          trusted.root,
          ["runs", configId, "result.json"],
          hashes.resultSha256,
          `${configId}/result.json`,
        ),
        readHashedArtifact(
          trusted.root,
          ["runs", configId, "receipt.json"],
          hashes.receiptSha256,
          `${configId}/receipt.json`,
        ),
        readHashedArtifact(
          trusted.root,
          ["runs", configId, "events.jsonl"],
          hashes.eventsSha256,
          `${configId}/events.jsonl`,
        ),
        readHashedArtifact(
          trusted.root,
          ["runs", configId, "stderr.log"],
          hashes.stderrSha256,
          `${configId}/stderr.log`,
        ),
      ]);
      outputs[configId] = validateResult(
        parseJson(resultBytes, `${configId}/result.json`),
        expectedFrameIds,
        configId,
      );
      validateReceipt(
        parseJson(receiptBytes, `${configId}/receipt.json`),
        configId,
        hashes.resultSha256,
        trusted.report.configurations[configId],
      );
    }),
  );

  await Promise.all(
    pack.frames.map((frame) => readAndValidateImage(trusted.root, frame)),
  );

  return {
    schemaVersion: 1,
    benchmarkId,
    width: pack.width,
    height: pack.height,
    policy: pack.policy,
    reportSha256: trusted.sha256,
    sample: trusted.report.sample,
    frames: pack.frames.map((frame) => {
      const frameOutputs = {} as Record<
        BenchmarkConfigurationId,
        BenchmarkFrameAnnotation
      >;
      for (const id of benchmarkConfigurationIds) {
        frameOutputs[id] = outputs[id][frame.frameId];
      }
      return {
        attachmentIndex: frame.attachmentIndex,
        frameId: frame.frameId,
        imageUrl: `/api/ball-review-benchmark/images/${frame.attachmentIndex}`,
        reference: reference[frame.frameId],
        outputs: frameOutputs,
      };
    }),
    configurations: trusted.report.configurations,
    pairwise: trusted.report.pairwise,
    metricSemantics: trusted.report.metricSemantics,
    limitations: trusted.report.limitations,
  };
}

export async function getBallReviewBenchmarkImage(
  attachmentIndex: number,
): Promise<BallReviewBenchmarkImage> {
  if (
    !Number.isSafeInteger(attachmentIndex) ||
    attachmentIndex < 1 ||
    attachmentIndex > frameCount
  ) {
    throw new BallReviewBenchmarkNotFoundError();
  }
  const trusted = await loadTrustedReport();
  const pack = await loadBlindPack(trusted);
  const frame = pack.frames.find(
    (candidate) => candidate.attachmentIndex === attachmentIndex,
  );
  if (!frame) throw new BallReviewBenchmarkNotFoundError();
  return readAndValidateImage(trusted.root, frame);
}
