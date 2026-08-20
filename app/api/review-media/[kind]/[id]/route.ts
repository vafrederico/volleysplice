import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { Readable } from "node:stream";

import {
  resolveReviewMedia,
  ReviewMediaNotFoundError,
  type ReviewMediaKind,
} from "@/lib/server/review-media";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

type ByteRange = { start: number; end: number };

function parseByteRange(value: string, size: number): ByteRange | null {
  if (!value.startsWith("bytes=") || value.includes(",")) return null;
  const match = /^bytes=(\d*)-(\d*)$/.exec(value);
  if (!match || (!match[1] && !match[2])) return null;
  if (!match[1]) {
    const suffixLength = Number(match[2]);
    if (!Number.isSafeInteger(suffixLength) || suffixLength <= 0) return null;
    return { start: Math.max(0, size - suffixLength), end: size - 1 };
  }
  const start = Number(match[1]);
  const requestedEnd = match[2] ? Number(match[2]) : size - 1;
  if (
    !Number.isSafeInteger(start) ||
    !Number.isSafeInteger(requestedEnd) ||
    start < 0 ||
    start >= size ||
    requestedEnd < start
  )
    return null;
  return { start, end: Math.min(requestedEnd, size - 1) };
}

function streamFile(
  filePath: string,
  request: Request,
  start?: number,
  end?: number,
): ReadableStream {
  const nodeStream = createReadStream(filePath, { start, end });
  request.signal.addEventListener("abort", () => nodeStream.destroy(), {
    once: true,
  });
  return Readable.toWeb(nodeStream) as ReadableStream;
}

function headers(filename: string, length: number): Headers {
  return new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, no-store",
    "Content-Disposition": `inline; filename*=UTF-8''${encodeURIComponent(filename)}`,
    "Content-Length": String(length),
    "Content-Type": "video/mp4",
    "X-Content-Type-Options": "nosniff",
  });
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ kind: string; id: string }> },
) {
  try {
    const { kind, id } = await params;
    if (kind !== "side-switch" && kind !== "serving-side") {
      throw new ReviewMediaNotFoundError("review kind is invalid");
    }
    const media = await resolveReviewMedia(kind as ReviewMediaKind, id);
    const size = (await stat(media.path)).size;
    const rangeValue = request.headers.get("range");
    if (!rangeValue) {
      return new Response(streamFile(media.path, request), {
        status: 200,
        headers: headers(media.filename, size),
      });
    }
    const range = parseByteRange(rangeValue, size);
    if (!range) {
      return new Response(null, {
        status: 416,
        headers: {
          "Accept-Ranges": "bytes",
          "Content-Range": `bytes */${size}`,
        },
      });
    }
    const responseHeaders = headers(
      media.filename,
      range.end - range.start + 1,
    );
    responseHeaders.set(
      "Content-Range",
      `bytes ${range.start}-${range.end}/${size}`,
    );
    return new Response(
      streamFile(media.path, request, range.start, range.end),
      {
        status: 206,
        headers: responseHeaders,
      },
    );
  } catch (error) {
    if (error instanceof ReviewMediaNotFoundError) {
      return Response.json({ error: error.message }, { status: 404 });
    }
    return Response.json(
      { error: "Review video is unavailable" },
      { status: 503 },
    );
  }
}
