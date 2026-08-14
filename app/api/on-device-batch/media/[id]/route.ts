import {
  assertOnDeviceBatchAuthorized,
  ON_DEVICE_BATCH_RECORDING_IDS,
  OnDeviceBatchAuthorizationError,
  OnDeviceBatchConfigurationError,
} from "../../../../../lib/server/on-device-batch.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const recordingIds = new Set<string>(ON_DEVICE_BATCH_RECORDING_IDS);

function privateHeaders(): HeadersInit {
  return { "Cache-Control": "private, no-store", Vary: "Authorization" };
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    assertOnDeviceBatchAuthorized(request);
  } catch (error) {
    if (error instanceof OnDeviceBatchAuthorizationError) {
      return Response.json(
        { error: "Unauthorized" },
        {
          status: 401,
          headers: { ...privateHeaders(), "WWW-Authenticate": "Bearer" },
        },
      );
    }
    if (error instanceof OnDeviceBatchConfigurationError) {
      return Response.json(
        { error: "On-device batch is unavailable" },
        { status: 503, headers: privateHeaders() },
      );
    }
    throw error;
  }

  const { id } = await params;
  if (!recordingIds.has(id)) {
    return Response.json(
      { error: "Video is not in the fixed batch" },
      { status: 404, headers: privateHeaders() },
    );
  }
  const { GET: servePreparedProxy } = await import(
    "../../../labeling/tasks/[id]/video/route.ts"
  );
  const response = await servePreparedProxy(request, { params: Promise.resolve({ id }) });
  response.headers.set("Cache-Control", "private, no-store");
  response.headers.set("Vary", "Authorization");
  return response;
}
