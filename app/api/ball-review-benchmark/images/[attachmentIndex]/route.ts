import {
  BallReviewBenchmarkNotFoundError,
  BallReviewBenchmarkUnavailableError,
  BallReviewBenchmarkValidationError,
  getBallReviewBenchmarkImage,
} from "@/lib/server/ball-review-benchmark";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ attachmentIndex: string }> },
) {
  try {
    const { attachmentIndex: rawAttachmentIndex } = await params;
    if (!/^(?:[1-9]|1[0-2])$/.test(rawAttachmentIndex)) {
      throw new BallReviewBenchmarkNotFoundError();
    }
    const image = await getBallReviewBenchmarkImage(Number(rawAttachmentIndex));
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
    if (error instanceof BallReviewBenchmarkNotFoundError) {
      return Response.json(
        { error: "Benchmark image not found" },
        { status: 404, headers: { "Cache-Control": "private, no-store" } },
      );
    }
    if (error instanceof BallReviewBenchmarkValidationError) {
      return Response.json(
        { error: "Benchmark image failed provenance validation" },
        { status: 409, headers: { "Cache-Control": "private, no-store" } },
      );
    }
    if (!(error instanceof BallReviewBenchmarkUnavailableError)) {
      console.error("Unexpected ball benchmark image error", error);
    }
    return Response.json(
      { error: "Benchmark image is unavailable" },
      { status: 503, headers: { "Cache-Control": "private, no-store" } },
    );
  }
}
