import {
  getPreparedLabelingCatalog,
  getSavedLabelingDocument,
  type LabelingBatch,
} from "@/lib/server/labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const catalog = await getPreparedLabelingCatalog();
    const savedDocuments = await Promise.all(
      catalog.tasks.map((task) => getSavedLabelingDocument(task)),
    );
    const batches = Object.fromEntries(
      (["full", "pilot"] satisfies LabelingBatch[]).map((batch) => {
        const batchTasks = catalog.tasks.filter((task) => task.batch === batch);
        return [
          batch,
          {
            ready: batchTasks.length,
            total: catalog.totals[batch],
            saved: batchTasks.filter((task) => {
              const index = catalog.tasks.indexOf(task);
              return savedDocuments[index].savedAt !== null;
            }).length,
            prelabeled: batchTasks.filter((task) => {
              const index = catalog.tasks.indexOf(task);
              return ["production-model", "prelabel"].includes(
                savedDocuments[index].source,
              );
            }).length,
          },
        ];
      }),
    );
    return Response.json(
      {
        batches,
        tasks: catalog.tasks.map((task, index) => ({
          id: task.id,
          batch: task.batch,
          priority: task.priority,
          environment: task.document.recording.environment,
          split: task.document.recording.split,
          durationSeconds: task.document.recording.durationSeconds,
          originalFilename: task.originalFilename,
          videoFilename: task.document.recording.videoFilename,
          documentSource: savedDocuments[index].source,
          savedAt: savedDocuments[index].savedAt,
          annotationStatus: savedDocuments[index].document.annotation.status,
          rallyCount: savedDocuments[index].document.rallies.length,
        })),
      },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch {
    return Response.json(
      { error: "Prepared labeling tasks are unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
