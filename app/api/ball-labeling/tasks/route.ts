import {
  BallLabelingWorkspaceError,
  getBallLabelingCatalog,
} from "@/lib/server/ball-labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    return Response.json(
      { tasks: await getBallLabelingCatalog() },
      { headers: { "Cache-Control": "no-store" } },
    );
  } catch (error) {
    if (!(error instanceof BallLabelingWorkspaceError)) console.error(error);
    return Response.json(
      { error: "Ball-labeling tasks are unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
