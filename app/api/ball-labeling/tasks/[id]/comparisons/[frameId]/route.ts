import {
  BallComparisonAccessError,
  BallLabelingTaskNotFoundError,
  getBallComparisonLayers,
} from "@/lib/server/ball-labeling-tasks";
import type { BallProposalSource } from "@/lib/ball-annotations";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function POST(
  request: Request,
  { params }: { params: Promise<{ id: string; frameId: string }> },
) {
  if (!request.headers.get("content-type")?.toLowerCase().startsWith("application/json")) {
    return Response.json({ error: "Content-Type must be application/json" }, { status: 415 });
  }
  try {
    const { id, frameId } = await params;
    const body = (await request.json()) as {
      mode?: unknown;
      sources?: unknown;
    };
    if (
      !["post-decision", "assisted"].includes(String(body.mode)) ||
      !Array.isArray(body.sources)
    ) {
      throw new BallComparisonAccessError("comparison request is invalid");
    }
    const result = await getBallComparisonLayers(
      id,
      frameId,
      body.mode as "post-decision" | "assisted",
      body.sources as BallProposalSource[],
    );
    return Response.json(result, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (error instanceof BallLabelingTaskNotFoundError) {
      return Response.json({ error: "Frame not found" }, { status: 404 });
    }
    if (error instanceof BallComparisonAccessError || error instanceof SyntaxError) {
      return Response.json(
        { error: error instanceof Error ? error.message : "Comparison request is invalid" },
        { status: 409, headers: { "Cache-Control": "no-store" } },
      );
    }
    return Response.json(
      { error: "Comparison layers are unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
