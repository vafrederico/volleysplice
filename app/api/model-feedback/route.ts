import { ModelFeedbackValidationError } from "../../../lib/model-feedback.ts";
import {
  listModelFeedbackImports,
  ModelFeedbackSourceMismatchError,
  ModelFeedbackStoreError,
  saveModelFeedbackImport,
} from "../../../lib/server/model-feedback-store.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_BODY_BYTES = 128 * 1024 * 1024;

function headers(): HeadersInit {
  return { "Cache-Control": "private, no-store" };
}

function publicOrigin(request: Request): string | null {
  const requestUrl = new URL(request.url);
  const forwardedHost = request.headers
    .get("x-forwarded-host")
    ?.split(",", 1)[0]
    .trim();
  const forwardedProtocol = request.headers
    .get("x-forwarded-proto")
    ?.split(",", 1)[0]
    .trim();
  const host =
    forwardedHost || request.headers.get("host")?.trim() || requestUrl.host;
  const protocol = forwardedProtocol || requestUrl.protocol.slice(0, -1);
  if (!host || (protocol !== "http" && protocol !== "https")) return null;
  try {
    return new URL(`${protocol}://${host}`).origin;
  } catch {
    return null;
  }
}

function errorResponse(error: unknown): Response {
  if (error instanceof ModelFeedbackValidationError) {
    return Response.json(
      { error: error.message },
      { status: 400, headers: headers() },
    );
  }
  if (error instanceof ModelFeedbackSourceMismatchError) {
    return Response.json(
      { error: error.message },
      { status: 422, headers: headers() },
    );
  }
  if (error instanceof ModelFeedbackStoreError) {
    return Response.json(
      { error: error.message },
      { status: 503, headers: headers() },
    );
  }
  return Response.json(
    { error: "The model-feedback import could not be saved" },
    { status: 503, headers: headers() },
  );
}

export async function GET() {
  try {
    return Response.json(
      { schemaVersion: 1, imports: await listModelFeedbackImports() },
      { headers: headers() },
    );
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    const origin = request.headers.get("origin");
    if (origin === null || origin !== publicOrigin(request)) {
      return Response.json(
        { error: "Origin does not match the application" },
        { status: 403, headers: headers() },
      );
    }
    const declaredLength = Number(request.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > MAX_BODY_BYTES) {
      return Response.json(
        { error: "The model-feedback file is too large" },
        { status: 413, headers: headers() },
      );
    }
    if (
      !request.headers.get("content-type")?.startsWith("multipart/form-data;")
    ) {
      return Response.json(
        { error: "Content-Type must be multipart/form-data" },
        { status: 415, headers: headers() },
      );
    }
    const form = await request.formData();
    const bundle = form.get("bundle");
    const sourcePath = form.get("sourcePath");
    if (!(bundle instanceof File)) {
      return Response.json(
        { error: "A model-feedback JSON file is required" },
        { status: 400, headers: headers() },
      );
    }
    if (sourcePath !== null && typeof sourcePath !== "string") {
      return Response.json(
        { error: "The server source path must be text" },
        { status: 400, headers: headers() },
      );
    }
    if (bundle.size > MAX_BODY_BYTES) {
      return Response.json(
        { error: "The model-feedback file is too large" },
        { status: 413, headers: headers() },
      );
    }
    const saved = await saveModelFeedbackImport(
      await bundle.text(),
      sourcePath?.trim() || null,
    );
    return Response.json(
      { created: true, import: saved },
      { status: 201, headers: headers() },
    );
  } catch (error) {
    return errorResponse(error);
  }
}
