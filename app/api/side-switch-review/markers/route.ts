import {
  loadFullVideoSideSwitchMarkerState,
  SideSwitchReviewStoreError,
  SideSwitchReviewValidationError,
  saveFullVideoSideSwitchMarkers,
} from "../../../../lib/server/side-switch-review.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const MAX_BODY_BYTES = 256 * 1024;

function headers(): HeadersInit {
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
  if (error instanceof SideSwitchReviewValidationError) {
    return Response.json(
      { error: error.message },
      { status: 400, headers: headers() },
    );
  }
  if (error instanceof SideSwitchReviewStoreError) {
    return Response.json(
      { error: error.message },
      { status: 503, headers: headers() },
    );
  }
  return Response.json(
    { error: "The full-video side-switch markers could not be saved" },
    { status: 503, headers: headers() },
  );
}

export async function GET() {
  try {
    return Response.json(await loadFullVideoSideSwitchMarkerState(), {
      headers: headers(),
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
        { status: 403, headers: headers() },
      );
    }
    if (
      request.headers.get("content-type")?.split(";", 1)[0].trim() !==
      "application/json"
    ) {
      return Response.json(
        { error: "Content-Type must be application/json" },
        { status: 415, headers: headers() },
      );
    }
    const declaredLength = Number(request.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > MAX_BODY_BYTES) {
      return Response.json(
        { error: "Request body is too large" },
        { status: 413, headers: headers() },
      );
    }
    const text = await request.text();
    if (Buffer.byteLength(text, "utf8") > MAX_BODY_BYTES) {
      return Response.json(
        { error: "Request body is too large" },
        { status: 413, headers: headers() },
      );
    }
    let body: unknown;
    try {
      body = JSON.parse(text) as unknown;
    } catch {
      return Response.json(
        { error: "Request body must be valid JSON" },
        { status: 400, headers: headers() },
      );
    }
    return Response.json(await saveFullVideoSideSwitchMarkers(body), {
      headers: headers(),
    });
  } catch (error) {
    return errorResponse(error);
  }
}
