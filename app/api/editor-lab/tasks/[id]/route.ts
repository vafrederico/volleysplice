import { getPreparedLabelingTask, getSavedLabelingDocument, LabelingTaskNotFoundError } from "@/lib/server/labeling-tasks";
import { humanExportConfiguration, loadProductionEditorLab } from "@/lib/server/production-editor-lab";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  try {
    const { id } = await params;
    const prepared = await getPreparedLabelingTask(id);
    // Model configurations remain label blind. Human labels are a separate editable reference.
    const task = await loadProductionEditorLab(prepared.document.recording);
    if (!task) return Response.json({ error: "No editor lab model predictions are prepared for this recording." },
      { status: 404, headers: { "Cache-Control": "no-store" } });
    const saved = await getSavedLabelingDocument(prepared);
    task.ignoredIntervals = saved.document.ignoredIntervals.map(({ start, end }) => ({ start, end }));
    const imported = prepared.corpusRecord?.humanReviewedImport || ["human-reviewed-model-feedback-export", "imported-human-labels"].includes(String(prepared.document.recording.capture.sourceType));
    if (saved.source === "draft" || saved.source === "completed" || imported) {
      task.configurations.push(humanExportConfiguration(saved.document, saved.source === "draft" || saved.source === "completed" ? saved.source : "imported"));
      task.provenance.labelBlindScope = "model-configurations";
    }
    return Response.json(task, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (error instanceof LabelingTaskNotFoundError) return Response.json({ error: "Task not found" }, { status: 404 });
    console.error("Editor lab recording could not be loaded", error);
    return Response.json({ error: "The editor lab model predictions are unavailable or do not match this recording." },
      { status: 503, headers: { "Cache-Control": "no-store" } });
  }
}
