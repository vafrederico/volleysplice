import { listPreparedLabelingTasks } from "@/lib/server/labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const tasks = await listPreparedLabelingTasks();
    return Response.json(
      {
        tasks: tasks.map((task) => ({
          id: task.id,
          priority: task.priority,
          environment: task.document.recording.environment,
          split: task.document.recording.split,
          durationSeconds: task.document.recording.durationSeconds,
          originalFilename: task.originalFilename,
          videoFilename: task.document.recording.videoFilename,
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
