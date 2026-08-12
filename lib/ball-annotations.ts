export const ballAnnotationTaskType = "volleycut-ball-presence-frame-annotation";

export const primaryBallStates = [
  "localizable",
  "fully_occluded",
  "out_of_frame",
  "indeterminate",
] as const;

export const ballObjectRoles = ["primary-court", "other-court", "unknown"] as const;
export const ballObjectVisibilities = [
  "clear",
  "motion-blurred",
  "partially-occluded",
] as const;

export type PrimaryBallState = (typeof primaryBallStates)[number];
export type BallObjectRole = (typeof ballObjectRoles)[number];
export type BallObjectVisibility = (typeof ballObjectVisibilities)[number];

export type NormalizedBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type BallObject = {
  id: string;
  category: "volleyball";
  role: BallObjectRole;
  bbox: NormalizedBox;
  visibility: BallObjectVisibility;
  truncated: boolean;
};

export type BallFrameAnnotation = {
  status: "unreviewed" | "reviewed";
  primaryBallState: PrimaryBallState | null;
  objects: BallObject[];
  notes: string;
  proposalExposure: "not_shown" | "shown_before_label_finalized";
  proposalSources: BallProposalSource[];
};

export function copyBallFrameLabelPreservingExposure(
  source: BallFrameAnnotation,
  target: BallFrameAnnotation,
): BallFrameAnnotation {
  return {
    status: source.status,
    primaryBallState: source.primaryBallState,
    objects: source.objects.map((object) => ({
      ...object,
      bbox: { ...object.bbox },
    })),
    notes: source.notes,
    proposalExposure: target.proposalExposure,
    proposalSources: [...(target.proposalSources ?? [])],
  };
}

export type BallFrame = {
  id: string;
  windowId: string;
  sampleOffset: number;
  sourceFrameIndex: number;
  sourceTimestampSeconds: number;
  image: {
    path: string;
    sha256: string;
    width: number;
    height: number;
    format: "png";
  };
};

export type BallWindow = {
  id: string;
  requestedStratum: string;
  actualSource: string;
  startSampleIndex: number;
  startSeconds: number;
  endSeconds: number;
  centerSeconds: number;
  reference: Record<string, unknown>;
  fallbackReason?: string;
};

export type BallImmutable = {
  taskId: string;
  digestSha256: string;
  manifest: Record<string, unknown>;
  recording: {
    id: string;
    split: "train" | "validation";
    sourceGroup: string;
    environment: string;
  };
  source: {
    proxy: {
      filename: string;
      pathHint: string;
      sizeBytes: number;
      sha256: string;
      width: number;
      height: number;
      fps: number;
      frameCount: number;
      durationSeconds: number;
    };
    normalizationProvenance: unknown;
  };
  sampling: Record<string, unknown>;
  annotationPolicy: Record<string, unknown>;
  windows: BallWindow[];
  frames: BallFrame[];
};

export type BallReview = {
  status: "unreviewed" | "in_progress" | "complete";
  annotator: string | null;
  reviewedAt: string | null;
  notes: string;
};

export type BallReviewDocument = {
  schemaVersion: 1;
  taskType: typeof ballAnnotationTaskType;
  immutable: BallImmutable;
  annotations: {
    review: BallReview;
    frames: Record<string, BallFrameAnnotation>;
  };
};

export type BallAnnotationTask = BallReviewDocument & {
  suggestions: {
    status: "empty";
    model: null;
    frames: Record<string, never>;
  };
};

export type BallProposalSource = "sol" | "detector";
export type BallProposalExposure = "blind" | "sol" | "detector" | "both";

export type BallDetectorVariantId =
  | "full-frame-v1"
  | "full-plus-overlap-2x2-v1";

export type BallDetectorComparisonLayer = {
  provenance: {
    artifactSha256: string;
    modelId: string;
    modelSha256: string | null;
    variantId: BallDetectorVariantId;
    label: string;
  };
  ballPresenceProbability: number;
  detections: Array<{ confidence: number; bbox: NormalizedBox }>;
};

export type BallComparisonLayers = {
  schemaVersion: 1;
  taskId: string;
  frameId: string;
  accessMode: "post-decision" | "assisted";
  proposalExposure: BallProposalExposure;
  layers: {
    sol: null | {
      provenance: {
        artifactSha256: string;
        annotator: string;
        reviewedAt: string;
      };
      annotation: BallFrameAnnotation;
    };
    detector: BallDetectorComparisonLayer | null;
    detectorTiled: BallDetectorComparisonLayer | null;
  };
};

export class BallReviewValidationError extends Error {}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(value: Record<string, unknown>, expected: string[]): boolean {
  const keys = Object.keys(value).sort();
  return keys.length === expected.length && keys.every((key, index) => key === [...expected].sort()[index]);
}

function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function positiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && typeof value === "number" && value > 0;
}

function validateBox(value: unknown, where: string): asserts value is NormalizedBox {
  if (!isRecord(value) || !hasExactKeys(value, ["x", "y", "width", "height"])) {
    throw new BallReviewValidationError(`${where} must be a normalized rectangle`);
  }
  const coordinates = [value.x, value.y, value.width, value.height];
  if (!coordinates.every(finiteNumber)) {
    throw new BallReviewValidationError(`${where} coordinates must be finite numbers`);
  }
  const { x, y, width, height } = value as NormalizedBox;
  if (x < 0 || y < 0 || width <= 0 || height <= 0 || x + width > 1 || y + height > 1) {
    throw new BallReviewValidationError(`${where} must fit inside the image`);
  }
}

function validateImmutable(value: unknown): asserts value is BallImmutable {
  if (!isRecord(value)) throw new BallReviewValidationError("immutable must be an object");
  if (
    typeof value.taskId !== "string" ||
    typeof value.digestSha256 !== "string" ||
    !/^[a-f0-9]{64}$/.test(value.digestSha256)
  ) {
    throw new BallReviewValidationError("immutable task identity is invalid");
  }
  if (!isRecord(value.recording) || typeof value.recording.id !== "string") {
    throw new BallReviewValidationError("immutable recording identity is invalid");
  }
  if (!isRecord(value.source) || !isRecord(value.source.proxy)) {
    throw new BallReviewValidationError("immutable proxy metadata is missing");
  }
  const proxy = value.source.proxy;
  if (!positiveInteger(proxy.width) || !positiveInteger(proxy.height)) {
    throw new BallReviewValidationError("immutable proxy dimensions are invalid");
  }
  if (!Array.isArray(value.windows) || value.windows.length === 0) {
    throw new BallReviewValidationError("immutable windows are missing");
  }
  const windowIds = new Set<string>();
  value.windows.forEach((candidate, index) => {
    if (!isRecord(candidate) || typeof candidate.id !== "string" || windowIds.has(candidate.id)) {
      throw new BallReviewValidationError(`immutable.windows[${index}] has an invalid id`);
    }
    windowIds.add(candidate.id);
  });
  if (!Array.isArray(value.frames) || value.frames.length === 0) {
    throw new BallReviewValidationError("immutable frames are missing");
  }
  const frameIds = new Set<string>();
  value.frames.forEach((candidate, index) => {
    const where = `immutable.frames[${index}]`;
    if (
      !isRecord(candidate) ||
      typeof candidate.id !== "string" ||
      frameIds.has(candidate.id) ||
      typeof candidate.windowId !== "string" ||
      !windowIds.has(candidate.windowId) ||
      !Number.isInteger(candidate.sampleOffset) ||
      !Number.isInteger(candidate.sourceFrameIndex) ||
      !finiteNumber(candidate.sourceTimestampSeconds) ||
      !isRecord(candidate.image)
    ) {
      throw new BallReviewValidationError(`${where} is invalid`);
    }
    const image = candidate.image;
    if (
      typeof image.path !== "string" ||
      !/^[a-f0-9]{64}$/.test(String(image.sha256)) ||
      image.width !== proxy.width ||
      image.height !== proxy.height ||
      image.format !== "png"
    ) {
      throw new BallReviewValidationError(`${where}.image is invalid`);
    }
    frameIds.add(candidate.id);
  });
}

function validateAnnotations(
  value: unknown,
  frames: BallFrame[],
): asserts value is BallReviewDocument["annotations"] {
  if (!isRecord(value) || !hasExactKeys(value, ["review", "frames"])) {
    throw new BallReviewValidationError("annotations must contain review and frames");
  }
  if (!isRecord(value.review) || !hasExactKeys(value.review, ["status", "annotator", "reviewedAt", "notes"])) {
    throw new BallReviewValidationError("annotations.review is invalid");
  }
  const review = value.review;
  if (
    !["unreviewed", "in_progress", "complete"].includes(String(review.status)) ||
    (review.annotator !== null && typeof review.annotator !== "string") ||
    (review.reviewedAt !== null && typeof review.reviewedAt !== "string") ||
    typeof review.notes !== "string"
  ) {
    throw new BallReviewValidationError("annotations.review metadata is invalid");
  }
  if (!isRecord(value.frames)) {
    throw new BallReviewValidationError("annotations.frames must be an object");
  }
  const expectedIds = new Set(frames.map((frame) => frame.id));
  if (
    Object.keys(value.frames).length !== expectedIds.size ||
    Object.keys(value.frames).some((frameId) => !expectedIds.has(frameId))
  ) {
    throw new BallReviewValidationError("annotations.frames must match immutable frame ids exactly");
  }

  let reviewedCount = 0;
  for (const [frameId, annotationValue] of Object.entries(value.frames)) {
    const where = `annotations.frames[${frameId}]`;
    const annotationKeys = isRecord(annotationValue) ? Object.keys(annotationValue) : [];
    if (
      !isRecord(annotationValue) ||
      !["status", "primaryBallState", "objects", "notes"].every((key) => annotationKeys.includes(key)) ||
      annotationKeys.some((key) =>
        ![
          "status",
          "primaryBallState",
          "objects",
          "notes",
          "proposalExposure",
          "proposalSources",
        ].includes(key),
      ) ||
      !["unreviewed", "reviewed"].includes(String(annotationValue.status)) ||
      !Array.isArray(annotationValue.objects) ||
      typeof annotationValue.notes !== "string" ||
      (annotationValue.proposalExposure !== undefined &&
        !["not_shown", "shown_before_label_finalized"].includes(
          String(annotationValue.proposalExposure),
        )) ||
      (annotationValue.proposalSources !== undefined &&
        (!Array.isArray(annotationValue.proposalSources) ||
          annotationValue.proposalSources.some(
            (source) => source !== "sol" && source !== "detector",
          ) ||
          new Set(annotationValue.proposalSources).size !==
            annotationValue.proposalSources.length ||
          (annotationValue.proposalExposure === "not_shown" &&
            annotationValue.proposalSources.length > 0)))
    ) {
      throw new BallReviewValidationError(`${where} is invalid`);
    }
    const annotation = annotationValue as BallFrameAnnotation;
    if (annotation.status === "unreviewed") {
      if (annotation.primaryBallState !== null || annotation.objects.length !== 0) {
        throw new BallReviewValidationError(`${where} has labels before review`);
      }
      continue;
    }
    reviewedCount += 1;
    if (!primaryBallStates.includes(annotation.primaryBallState as PrimaryBallState)) {
      throw new BallReviewValidationError(`${where}.primaryBallState is invalid`);
    }
    const objectIds = new Set<string>();
    let primaryCount = 0;
    annotation.objects.forEach((objectValue, objectIndex) => {
      const objectWhere = `${where}.objects[${objectIndex}]`;
      if (
        !isRecord(objectValue) ||
        !hasExactKeys(objectValue, ["id", "category", "role", "bbox", "visibility", "truncated"]) ||
        typeof objectValue.id !== "string" ||
        !objectValue.id.trim() ||
        objectIds.has(objectValue.id) ||
        objectValue.category !== "volleyball" ||
        !ballObjectRoles.includes(objectValue.role as BallObjectRole) ||
        !ballObjectVisibilities.includes(objectValue.visibility as BallObjectVisibility) ||
        typeof objectValue.truncated !== "boolean"
      ) {
        throw new BallReviewValidationError(`${objectWhere} is invalid`);
      }
      validateBox(objectValue.bbox, `${objectWhere}.bbox`);
      objectIds.add(objectValue.id);
      if (objectValue.role === "primary-court") primaryCount += 1;
    });
    if (annotation.primaryBallState === "localizable" && primaryCount !== 1) {
      throw new BallReviewValidationError(`${where} requires exactly one primary-court ball`);
    }
    if (annotation.primaryBallState !== "localizable" && primaryCount !== 0) {
      throw new BallReviewValidationError(`${where} cannot contain a primary-court ball`);
    }
  }

  if (review.status === "unreviewed") {
    if (reviewedCount !== 0 || review.annotator !== null || review.reviewedAt !== null) {
      throw new BallReviewValidationError("unreviewed task metadata is inconsistent");
    }
  } else if (review.status === "in_progress") {
    if (
      reviewedCount <= 0 ||
      reviewedCount >= frames.length ||
      (review.annotator !== null && typeof review.annotator !== "string") ||
      review.reviewedAt !== null
    ) {
      throw new BallReviewValidationError("in-progress review metadata is inconsistent");
    }
  } else if (
    reviewedCount !== frames.length ||
    (review.annotator !== null && typeof review.annotator !== "string") ||
    typeof review.reviewedAt !== "string" ||
    !review.reviewedAt.trim() ||
    Number.isNaN(Date.parse(review.reviewedAt))
  ) {
    throw new BallReviewValidationError("complete review metadata is inconsistent");
  }
}

function parseReviewRoot(value: unknown): BallReviewDocument {
  if (!isRecord(value) || !hasExactKeys(value, ["schemaVersion", "taskType", "immutable", "annotations"])) {
    throw new BallReviewValidationError(
      "review submission contains unsupported fields (detector suggestions are forbidden)",
    );
  }
  if (value.schemaVersion !== 1 || value.taskType !== ballAnnotationTaskType) {
    throw new BallReviewValidationError("not a ball-presence annotation document v1");
  }
  validateImmutable(value.immutable);
  validateAnnotations(value.annotations, value.immutable.frames);
  const document = value as BallReviewDocument;
  return {
    ...document,
    annotations: {
      ...document.annotations,
      frames: Object.fromEntries(
        Object.entries(document.annotations.frames).map(([frameId, annotation]) => [
          frameId,
          {
            ...annotation,
            proposalExposure: annotation.proposalExposure ?? "not_shown",
            proposalSources: annotation.proposalSources ?? [],
          },
        ]),
      ),
    },
  };
}

export function parseBallReviewDocument(value: unknown): BallReviewDocument {
  return parseReviewRoot(value);
}

export function parseBallAnnotationTask(value: unknown): BallAnnotationTask {
  if (!isRecord(value) || !hasExactKeys(value, ["schemaVersion", "taskType", "immutable", "suggestions", "annotations"])) {
    throw new BallReviewValidationError("ball annotation task has unsupported fields");
  }
  const suggestions = value.suggestions;
  if (
    !isRecord(suggestions) ||
    !hasExactKeys(suggestions, ["status", "model", "frames"]) ||
    suggestions.status !== "empty" ||
    suggestions.model !== null ||
    !isRecord(suggestions.frames) ||
    Object.keys(suggestions.frames).length !== 0
  ) {
    throw new BallReviewValidationError(
      "human-review tasks must not contain detector suggestions",
    );
  }
  return {
    ...parseReviewRoot({
      schemaVersion: value.schemaVersion,
      taskType: value.taskType,
      immutable: value.immutable,
      annotations: value.annotations,
    }),
    suggestions: suggestions as BallAnnotationTask["suggestions"],
  };
}

export function toBallReviewDocument(task: BallAnnotationTask): BallReviewDocument {
  return {
    schemaVersion: task.schemaVersion,
    taskType: task.taskType,
    immutable: task.immutable,
    annotations: {
      review: { ...task.annotations.review },
      frames: Object.fromEntries(
        Object.entries(task.annotations.frames).map(([frameId, annotation]) => [
          frameId,
          {
            ...annotation,
            objects: annotation.objects.map((object) => ({
              ...object,
              bbox: { ...object.bbox },
            })),
          },
        ]),
      ),
    },
  };
}

export function reviewedFrameCount(document: BallReviewDocument): number {
  return Object.values(document.annotations.frames).filter((frame) => frame.status === "reviewed").length;
}
