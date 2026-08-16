import {
  ModelFeedbackImportNotFoundError,
  readModelFeedbackBundle,
} from "../../../../lib/server/model-feedback-store.ts";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const stored = await readModelFeedbackBundle(id);
    const headers = new Headers({
      "Cache-Control": "private, no-store",
      "Content-Type": "application/json; charset=utf-8",
    });
    if (stored.metadata.sourcePath && stored.metadata.mediaUrl) {
      headers.set(
        "X-VolleyCut-Source-Path",
        encodeURIComponent(stored.metadata.sourcePath),
      );
      headers.set("X-VolleyCut-Source-Url", stored.metadata.mediaUrl);
    }
    return new Response(stored.text, {
      headers,
    });
  } catch (error) {
    if (error instanceof ModelFeedbackImportNotFoundError) {
      return Response.json({ error: "Import not found" }, { status: 404 });
    }
    return Response.json(
      { error: "The stored import is unavailable" },
      { status: 503 },
    );
  }
}
