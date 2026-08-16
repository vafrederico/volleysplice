import { createHash } from "node:crypto";
import {
  lstat,
  mkdir,
  mkdtemp,
  open,
  readdir,
  readFile,
  realpath,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";

import { parseModelFeedbackText } from "../model-feedback.ts";
import { getModelFeedbackRoot } from "../storage.ts";

const IMPORT_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}$/;
const SAMPLE_BYTES = 1024 * 1024;

export type ModelFeedbackImportSummary = {
  id: string;
  importedAt: string;
  generatedAt: string;
  producer: "production-web" | "android";
  projectId: string;
  analysisId: string;
  sourceName: string;
  duration: number;
  modelId: string;
  featureRows: number | null;
  correctedRanges: number;
  sourceLinked: boolean;
  mediaUrl: string | null;
};

type StoredModelFeedbackImport = Omit<
  ModelFeedbackImportSummary,
  "sourceLinked" | "mediaUrl"
> & {
  bundlePath: string;
  sourcePath: string | null;
  sourceSize: number | null;
  sourceMimeType: string | null;
  sampledFingerprint: string | null;
};

export class ModelFeedbackStoreError extends Error {}
export class ModelFeedbackSourceMismatchError extends ModelFeedbackStoreError {}
export class ModelFeedbackImportNotFoundError extends ModelFeedbackStoreError {}

function safeSlug(value: string): string {
  return (
    value
      .replace(/[^A-Za-z0-9_-]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 54) || "feedback"
  );
}

function rootPath(configuredRoot?: string): string {
  const root = path.resolve(configuredRoot ?? getModelFeedbackRoot());
  if (root === path.parse(root).root) {
    throw new ModelFeedbackStoreError(
      "The model-feedback store cannot be a filesystem root",
    );
  }
  return root;
}

function metadataPath(root: string, id: string): string {
  if (!IMPORT_ID.test(id))
    throw new ModelFeedbackImportNotFoundError("Import not found");
  return path.join(root, id, "import.json");
}

function metadataValue(
  value: unknown,
  root: string,
): StoredModelFeedbackImport {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ModelFeedbackStoreError("Stored import metadata is invalid");
  }
  const item = value as Partial<StoredModelFeedbackImport>;
  const hasLinkedSource =
    typeof item.sourcePath === "string" &&
    path.isAbsolute(item.sourcePath) &&
    typeof item.sourceSize === "number" &&
    Number.isSafeInteger(item.sourceSize) &&
    item.sourceSize >= 0 &&
    typeof item.sourceMimeType === "string" &&
    (item.sampledFingerprint === null ||
      typeof item.sampledFingerprint === "string");
  const hasNoLinkedSource =
    item.sourcePath === null &&
    item.sourceSize === null &&
    item.sourceMimeType === null &&
    item.sampledFingerprint === null;
  if (
    typeof item.id !== "string" ||
    !IMPORT_ID.test(item.id) ||
    typeof item.importedAt !== "string" ||
    typeof item.generatedAt !== "string" ||
    (item.producer !== "production-web" && item.producer !== "android") ||
    typeof item.projectId !== "string" ||
    typeof item.analysisId !== "string" ||
    typeof item.sourceName !== "string" ||
    typeof item.duration !== "number" ||
    !Number.isFinite(item.duration) ||
    typeof item.modelId !== "string" ||
    (item.featureRows !== null && typeof item.featureRows !== "number") ||
    typeof item.correctedRanges !== "number" ||
    (!hasLinkedSource && !hasNoLinkedSource)
  ) {
    throw new ModelFeedbackStoreError("Stored import metadata is invalid");
  }
  return {
    ...item,
    bundlePath: path.join(root, item.id, "bundle.json"),
  } as StoredModelFeedbackImport;
}

async function readMetadata(
  root: string,
  id: string,
): Promise<StoredModelFeedbackImport> {
  let value: unknown;
  try {
    value = JSON.parse(
      await readFile(metadataPath(root, id), "utf8"),
    ) as unknown;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") {
      throw new ModelFeedbackImportNotFoundError("Import not found");
    }
    throw new ModelFeedbackStoreError(
      "Stored import metadata could not be read",
    );
  }
  const metadata = metadataValue(value, root);
  if (metadata.id !== id)
    throw new ModelFeedbackStoreError("Stored import ID does not match");
  return metadata;
}

export async function sampledSourceFingerprint(
  filePath: string,
  size?: number,
): Promise<string> {
  const sourceSize = size ?? (await stat(filePath)).size;
  const digest = createHash("sha256");
  const sizeBytes = Buffer.alloc(8);
  sizeBytes.writeBigUInt64LE(BigInt(sourceSize));
  digest.update(sizeBytes);
  const handle = await open(filePath, "r");
  try {
    const segments =
      sourceSize <= SAMPLE_BYTES * 2
        ? [{ position: 0, length: sourceSize }]
        : [
            { position: 0, length: SAMPLE_BYTES },
            { position: sourceSize - SAMPLE_BYTES, length: SAMPLE_BYTES },
          ];
    for (const segment of segments) {
      const bytes = Buffer.alloc(segment.length);
      const { bytesRead } = await handle.read(
        bytes,
        0,
        segment.length,
        segment.position,
      );
      if (bytesRead !== segment.length) {
        throw new ModelFeedbackStoreError(
          "The source ended before its declared size",
        );
      }
      digest.update(bytes);
    }
  } finally {
    await handle.close();
  }
  return `sampled-sha256-v1:${digest.digest("hex")}`;
}

async function validateSourcePath(
  requestedPath: string,
  expectedSize: number,
  expectedFingerprint: string | null,
): Promise<{ sourcePath: string; sourceSize: number }> {
  if (!path.isAbsolute(requestedPath)) {
    throw new ModelFeedbackSourceMismatchError(
      "The server source path must be absolute",
    );
  }
  let sourcePath: string;
  let sourceStats;
  try {
    sourcePath = await realpath(requestedPath);
    sourceStats = await lstat(sourcePath);
  } catch {
    throw new ModelFeedbackSourceMismatchError(
      "The server cannot read that source path",
    );
  }
  if (!sourceStats.isFile()) {
    throw new ModelFeedbackSourceMismatchError(
      "The server source path must point to a file",
    );
  }
  if (sourceStats.size !== expectedSize) {
    throw new ModelFeedbackSourceMismatchError(
      `The source is ${sourceStats.size} bytes, but the bundle expects ${expectedSize}`,
    );
  }
  if (expectedFingerprint) {
    const actual = await sampledSourceFingerprint(sourcePath, sourceStats.size);
    if (actual !== expectedFingerprint) {
      throw new ModelFeedbackSourceMismatchError(
        "The source fingerprint does not match the bundle",
      );
    }
  }
  return {
    sourcePath,
    sourceSize: sourceStats.size,
  };
}

function publicSummary(
  metadata: StoredModelFeedbackImport,
): ModelFeedbackImportSummary {
  const {
    bundlePath: _bundlePath,
    sourcePath: _sourcePath,
    sourceSize: _sourceSize,
    sourceMimeType: _sourceMimeType,
    sampledFingerprint: _sampledFingerprint,
    ...summary
  } = metadata;
  return {
    ...summary,
    sourceLinked: _sourcePath !== null,
    mediaUrl:
      _sourcePath === null
        ? null
        : `/api/model-feedback/${encodeURIComponent(metadata.id)}/source`,
  };
}

export async function saveModelFeedbackImport(
  bundleText: string,
  requestedSourcePath: string | null,
  options: { root?: string; importedAt?: string } = {},
): Promise<ModelFeedbackImportSummary> {
  const bundle = parseModelFeedbackText(bundleText);
  const linkedSource = requestedSourcePath
    ? await validateSourcePath(
        requestedSourcePath,
        bundle.source.file.sizeBytes,
        bundle.source.file.sampledFingerprint,
      )
    : { sourcePath: null, sourceSize: null };
  const { sourcePath, sourceSize } = linkedSource;
  const root = rootPath(options.root);
  await mkdir(root, { recursive: true });
  const identity = createHash("sha256")
    .update(bundleText)
    .update("\0")
    .update(sourcePath ?? "")
    .digest("hex")
    .slice(0, 16);
  const id = `${safeSlug(bundle.source.projectId)}-${identity}`;
  const importedAt = options.importedAt ?? new Date().toISOString();
  const metadata: StoredModelFeedbackImport = {
    id,
    importedAt,
    generatedAt: bundle.generatedAt,
    producer: bundle.producer,
    projectId: bundle.source.projectId,
    analysisId: bundle.source.analysisId,
    sourceName: bundle.source.file.name,
    sourcePath,
    duration: bundle.source.media.duration,
    modelId: bundle.initialInference.modelId,
    featureRows: bundle.features?.rows ?? null,
    correctedRanges: bundle.corrections.correctedRanges.length,
    bundlePath: path.join(root, id, "bundle.json"),
    sourceSize,
    sourceMimeType: sourcePath
      ? bundle.source.file.mimeType || bundle.source.media.mimeType
      : null,
    sampledFingerprint: sourcePath
      ? bundle.source.file.sampledFingerprint
      : null,
  };

  try {
    const existing = await readMetadata(root, id);
    return publicSummary(existing);
  } catch (error) {
    if (!(error instanceof ModelFeedbackImportNotFoundError)) throw error;
  }

  const staging = await mkdtemp(path.join(root, `.import-${id}-`));
  try {
    await writeFile(
      path.join(staging, "bundle.json"),
      bundleText.endsWith("\n") ? bundleText : `${bundleText}\n`,
      {
        encoding: "utf8",
        flag: "wx",
      },
    );
    const storedMetadata = {
      ...metadata,
      bundlePath: undefined,
    };
    await writeFile(
      path.join(staging, "import.json"),
      `${JSON.stringify(storedMetadata, null, 2)}\n`,
      {
        encoding: "utf8",
        flag: "wx",
      },
    );
    await rename(staging, path.join(root, id));
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    if (
      (error as NodeJS.ErrnoException).code === "EEXIST" ||
      (error as NodeJS.ErrnoException).code === "ENOTEMPTY"
    ) {
      return publicSummary(await readMetadata(root, id));
    }
    throw error;
  }
  return publicSummary(metadata);
}

export async function listModelFeedbackImports(
  options: { root?: string } = {},
): Promise<ModelFeedbackImportSummary[]> {
  const root = rootPath(options.root);
  let entries;
  try {
    entries = await readdir(root, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  const imports = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory() && IMPORT_ID.test(entry.name))
      .map((entry) => readMetadata(root, entry.name).catch(() => null)),
  );
  return imports
    .filter((item): item is StoredModelFeedbackImport => item !== null)
    .sort((left, right) => right.importedAt.localeCompare(left.importedAt))
    .map(publicSummary);
}

export async function getModelFeedbackImport(
  id: string,
  options: { root?: string } = {},
): Promise<StoredModelFeedbackImport> {
  return readMetadata(rootPath(options.root), id);
}

export async function readModelFeedbackBundle(
  id: string,
  options: { root?: string } = {},
): Promise<{
  metadata: ModelFeedbackImportSummary & { sourcePath: string | null };
  text: string;
}> {
  const stored = await getModelFeedbackImport(id, options);
  return {
    metadata: { ...publicSummary(stored), sourcePath: stored.sourcePath },
    text: await readFile(stored.bundlePath, "utf8"),
  };
}
