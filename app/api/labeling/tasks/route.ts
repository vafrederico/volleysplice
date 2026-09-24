import {
  getPreparedLabelingCatalog,
  getSavedLabelingDocument,
  type LabelingBatch,
} from "@/lib/server/labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function isModelSeeded(
  saved: Awaited<ReturnType<typeof getSavedLabelingDocument>>,
  humanReviewedImport = false,
): boolean {
  if (humanReviewedImport && saved.source === "task") return false;
  return (
    ["production-model", "prelabel"].includes(saved.source) ||
    saved.document.rallies.some((rally) => rally.tags.includes("ai-prelabel")) ||
    saved.document.serveMarkers.some((marker) => marker.origin === "model") ||
    saved.document.sideSwitches.some((marker) => marker.origin === "model")
  );
}

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
              return isModelSeeded(savedDocuments[index], task.corpusRecord?.humanReviewedImport);
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
          modelSeeded: isModelSeeded(savedDocuments[index], task.corpusRecord?.humanReviewedImport),
          humanReviewedImport: task.corpusRecord?.humanReviewedImport === true,
          sourceType:
            typeof task.document.recording.capture.sourceType === "string"
              ? task.document.recording.capture.sourceType
              : null,
          targetStatus:
            typeof task.document.recording.capture.targetStatus === "string"
              ? task.document.recording.capture.targetStatus
              : null,
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
