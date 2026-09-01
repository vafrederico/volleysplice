import {
  getPreparedLabelingTask,
  LabelingDraftValidationError,
  LabelingTaskNotFoundError,
  saveCompletedLabelingDocument,
} from "@/lib/server/labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const maximumDocumentBytes = 1024 * 1024;

export async function PUT(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  if (
    !request.headers
      .get("content-type")
      ?.toLowerCase()
      .startsWith("application/json")
  ) {
    return Response.json(
      { error: "Content-Type must be application/json" },
      { status: 415 },
    );
  }
  const declaredLength = Number(request.headers.get("content-length"));
  if (
    Number.isFinite(declaredLength) &&
    declaredLength > maximumDocumentBytes
  ) {
    return Response.json(
      { error: "Label document is too large" },
      { status: 413 },
    );
  }
  try {
    const { id } = await params;
    const task = await getPreparedLabelingTask(id);
    const body = await request.text();
    if (new TextEncoder().encode(body).byteLength > maximumDocumentBytes) {
      return Response.json(
        { error: "Label document is too large" },
        { status: 413 },
      );
    }
    const saved = await saveCompletedLabelingDocument(
      task,
      JSON.parse(body) as unknown,
    );
    return Response.json(
      {
        saved: true,
        id: task.id,
        batch: task.batch,
        savedAt: saved.savedAt,
        rallies: saved.document.rallies.length,
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    if (error instanceof LabelingTaskNotFoundError) {
      return Response.json({ error: "Task not found" }, { status: 404 });
    }
    if (
      error instanceof LabelingDraftValidationError ||
      error instanceof SyntaxError
    ) {
      return Response.json(
        {
          error:
            error instanceof Error
              ? error.message
              : "Label document is invalid",
        },
        { status: 400 },
      );
    }
    return Response.json(
      { error: "Completed labels could not be saved" },
      { status: 503 },
    );
  }
}
