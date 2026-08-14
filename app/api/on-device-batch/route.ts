import {
  assertOnDeviceBatchAuthorized,
  getOnDeviceBatchCatalog,
  OnDeviceBatchAuthorizationError,
  OnDeviceBatchConfigurationError,
  OnDeviceBatchConflictError,
  OnDeviceBatchValidationError,
  saveOnDeviceBatchSubmission,
} from "../../../lib/server/on-device-batch.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_BODY_BYTES = 1024 * 1024;

function responseHeaders(): HeadersInit {
  return {
    "Cache-Control": "private, no-store",
    Vary: "Authorization",
  };
}

function firstForwardedValue(value: string | null): string | null {
  const first = value?.split(",", 1)[0].trim();
  return first || null;
}

function requestPublicOrigin(request: Request): string | null {
  const internal = new URL(request.url);
  const host =
    firstForwardedValue(request.headers.get("x-forwarded-host")) ??
    request.headers.get("host")?.trim() ??
    internal.host;
  const protocol =
    firstForwardedValue(request.headers.get("x-forwarded-proto")) ??
    internal.protocol.slice(0, -1);
  if (!host || (protocol !== "http" && protocol !== "https")) return null;
  try {
    return new URL(`${protocol}://${host}`).origin;
  } catch {
    return null;
  }
}

function errorResponse(error: unknown): Response {
  if (error instanceof OnDeviceBatchAuthorizationError) {
    return Response.json(
      { error: "Unauthorized" },
      {
        status: 401,
        headers: { ...responseHeaders(), "WWW-Authenticate": "Bearer" },
      },
    );
  }
  if (error instanceof OnDeviceBatchValidationError) {
    return Response.json({ error: error.message }, { status: 400, headers: responseHeaders() });
  }
  if (error instanceof OnDeviceBatchConflictError) {
    return Response.json(
      { error: "Analysis already exists" },
      { status: 409, headers: responseHeaders() },
    );
  }
  if (error instanceof OnDeviceBatchConfigurationError) {
    return Response.json(
      { error: "On-device batch is unavailable" },
      { status: 503, headers: responseHeaders() },
    );
  }
  return Response.json(
    { error: "On-device batch request failed" },
    { status: 503, headers: responseHeaders() },
  );
}

export async function GET(request: Request) {
  try {
    assertOnDeviceBatchAuthorized(request);
    return Response.json(await getOnDeviceBatchCatalog(), { headers: responseHeaders() });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    assertOnDeviceBatchAuthorized(request);
    const origin = request.headers.get("origin");
    if (origin === null || origin !== requestPublicOrigin(request)) {
      return Response.json(
        { error: "Origin does not match the application" },
        { status: 403, headers: responseHeaders() },
      );
    }
    const contentType = request.headers.get("content-type")?.split(";", 1)[0].trim();
    if (contentType !== "application/json") {
      return Response.json(
        { error: "Content-Type must be application/json" },
        { status: 415, headers: responseHeaders() },
      );
    }
    const declaredLength = Number(request.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > MAX_BODY_BYTES) {
      return Response.json(
        { error: "Request body is too large" },
        { status: 413, headers: responseHeaders() },
      );
    }
    const text = await request.text();
    if (Buffer.byteLength(text, "utf8") > MAX_BODY_BYTES) {
      return Response.json(
        { error: "Request body is too large" },
        { status: 413, headers: responseHeaders() },
      );
    }
    let body: unknown;
    try {
      body = JSON.parse(text) as unknown;
    } catch {
      throw new OnDeviceBatchValidationError("Request body must be valid JSON");
    }
    const saved = await saveOnDeviceBatchSubmission(body);
    return Response.json(
      { created: true, ...saved },
      { status: 201, headers: responseHeaders() },
    );
  } catch (error) {
    return errorResponse(error);
  }
}
