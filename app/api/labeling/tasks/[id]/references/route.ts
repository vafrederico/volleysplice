import {
  getPreparedLabelingTask,
  getSolReferenceLabels,
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
    return Response.json(
      { sol: await getSolReferenceLabels(task) },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    if (error instanceof LabelingTaskNotFoundError) {
      return Response.json({ error: "Task not found" }, { status: 404 });
    }
    return Response.json(
      { error: "Reference labels are unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
