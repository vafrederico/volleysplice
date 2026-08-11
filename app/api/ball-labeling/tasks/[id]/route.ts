import {
  BallLabelingDraftValidationError,
  BallLabelingTaskNotFoundError,
  getBallReviewDocument,
  saveBallReviewDocument,
} from "@/lib/server/ball-labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const maximumReviewBytes = 2 * 1024 * 1024;

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const saved = await getBallReviewDocument(id);
    return Response.json(saved.document, {
      headers: {
        "Cache-Control": "no-store",
        ...(saved.savedAt ? { "X-VolleyCut-Saved-At": saved.savedAt } : {}),
      },
    });
  } catch (error) {
    if (error instanceof BallLabelingTaskNotFoundError) {
      return Response.json({ error: "Task not found" }, { status: 404 });
    }
    return Response.json(
      { error: "Ball-labeling task is unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  if (!request.headers.get("content-type")?.toLowerCase().startsWith("application/json")) {
    return Response.json({ error: "Content-Type must be application/json" }, { status: 415 });
  }
  const declaredLength = Number(request.headers.get("content-length"));
  if (Number.isFinite(declaredLength) && declaredLength > maximumReviewBytes) {
    return Response.json({ error: "Review is too large" }, { status: 413 });
  }
  try {
    const { id } = await params;
    const body = await request.text();
    if (new TextEncoder().encode(body).byteLength > maximumReviewBytes) {
      return Response.json({ error: "Review is too large" }, { status: 413 });
    }
    const saved = await saveBallReviewDocument(id, JSON.parse(body) as unknown);
    return Response.json(
      {
        saved: true,
        id,
        savedAt: saved.savedAt,
        reviewStatus: saved.document.annotations.review.status,
        assistedFrameCount: Object.values(saved.document.annotations.frames).filter(
          (frame) => frame.proposalExposure === "shown_before_label_finalized",
        ).length,
        proposalExposureByFrame: Object.fromEntries(
          Object.entries(saved.document.annotations.frames).map(([frameId, frame]) => [
            frameId,
            frame.proposalExposure,
          ]),
        ),
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    if (error instanceof BallLabelingTaskNotFoundError) {
      return Response.json({ error: "Task not found" }, { status: 404 });
    }
    if (error instanceof BallLabelingDraftValidationError || error instanceof SyntaxError) {
      return Response.json(
        { error: error instanceof Error ? error.message : "Review is invalid" },
        { status: 400, headers: { "Cache-Control": "no-store" } },
      );
    }
    return Response.json(
      { error: "Review could not be saved" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
