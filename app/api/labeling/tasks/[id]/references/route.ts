import {
  getPreparedLabelingTask,
  getProductionReferenceLabels,
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
    const [production, sol] = await Promise.all([
      getProductionReferenceLabels(task),
      getSolReferenceLabels(task),
    ]);
    return Response.json(
      { production, sol },
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
