import {
  getPreparedLabelingTask,
  getSavedLabelingDocument,
  LabelingTaskNotFoundError,
} from "@/lib/server/labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const task = await getPreparedLabelingTask(id);
    const saved = await getSavedLabelingDocument(task);
    return Response.json(saved.document, {
      headers: {
        "Cache-Control": "no-store",
        "X-VolleyCut-Batch": task.batch,
        "X-VolleyCut-Document-Source": saved.source,
        ...(saved.savedAt ? { "X-VolleyCut-Saved-At": saved.savedAt } : {}),
      },
    });
  } catch (error) {
    if (error instanceof LabelingTaskNotFoundError) {
      return Response.json({ error: "Task not found" }, { status: 404 });
    }
    return Response.json(
      { error: "Prepared labeling task is unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
