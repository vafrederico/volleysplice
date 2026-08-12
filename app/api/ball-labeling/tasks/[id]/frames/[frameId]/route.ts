import {
  BallLabelingImageValidationError,
  BallLabelingTaskNotFoundError,
  getBallFrameImage,
} from "@/lib/server/ball-labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ id: string; frameId: string }> },
) {
  try {
    const { id, frameId } = await params;
    const image = await getBallFrameImage(id, frameId);
    const body = new ArrayBuffer(image.bytes.byteLength);
    new Uint8Array(body).set(image.bytes);
    return new Response(body, {
      headers: {
        "Cache-Control": "private, max-age=31536000, immutable",
        "Content-Disposition": `inline; filename="${image.filename}"`,
        "Content-Length": String(image.bytes.byteLength),
        "Content-Type": "image/png",
        ETag: `"${image.sha256}"`,
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch (error) {
    if (error instanceof BallLabelingTaskNotFoundError) {
      return Response.json({ error: "Frame not found" }, { status: 404 });
    }
    if (error instanceof BallLabelingImageValidationError) {
      return Response.json({ error: "Frame failed provenance validation" }, { status: 409 });
    }
    return Response.json({ error: "Frame is unavailable" }, { status: 503 });
  }
}
