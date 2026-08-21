import {
  loadServingSideFlightAnnotationState,
  ServingSideFlightReviewError,
  ServingSideFlightReviewValidationError,
  saveServingSideFlightAnnotation,
} from "../../../../lib/server/serving-side-flight-review.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_BODY_BYTES = 16 * 1024;

function responseHeaders(): HeadersInit {
  return { "Cache-Control": "private, no-store" };
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
  if (error instanceof ServingSideFlightReviewValidationError) {
    return Response.json(
      { error: error.message },
      { status: 400, headers: responseHeaders() },
    );
  }
  if (error instanceof ServingSideFlightReviewError) {
    return Response.json(
      { error: error.message },
      { status: 503, headers: responseHeaders() },
    );
  }
  return Response.json(
    { error: "The serving-side flight annotation could not be saved" },
    { status: 503, headers: responseHeaders() },
  );
}

export async function GET() {
  try {
    return Response.json(await loadServingSideFlightAnnotationState(), {
      headers: responseHeaders(),
    });
  } catch (error) {
    return errorResponse(error);
  }
}

export async function POST(request: Request) {
  try {
    const origin = request.headers.get("origin");
    if (origin === null || origin !== requestPublicOrigin(request)) {
      return Response.json(
        { error: "Origin does not match the application" },
        { status: 403, headers: responseHeaders() },
      );
    }
    if (
      request.headers.get("content-type")?.split(";", 1)[0].trim() !==
      "application/json"
    ) {
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
      return Response.json(
        { error: "Request body must be valid JSON" },
        { status: 400, headers: responseHeaders() },
      );
    }
    return Response.json(await saveServingSideFlightAnnotation(body), {
      headers: responseHeaders(),
    });
  } catch (error) {
    return errorResponse(error);
  }
}
