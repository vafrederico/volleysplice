import { createReadStream } from "node:fs";
import { Readable } from "node:stream";

import {
  BallLabelingTaskNotFoundError,
  BallLabelingVideoValidationError,
  getBallSourceVideo,
} from "@/lib/server/ball-labeling-tasks";

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
  ) {
    return null;
  }
  return { start, end: Math.min(requestedEnd, size - 1) };
}

function headers(filename: string, sha256: string, contentLength: number): Headers {
  return new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, max-age=31536000, immutable",
    "Content-Disposition": `inline; filename*=UTF-8''${encodeURIComponent(filename)}`,
    "Content-Length": String(contentLength),
    "Content-Type": "video/mp4",
    ETag: `"${sha256}"`,
    "X-Content-Type-Options": "nosniff",
  });
}

function stream(
  filePath: string,
  request: Request,
  start?: number,
  end?: number,
): ReadableStream {
  const source = createReadStream(filePath, { start, end });
  request.signal.addEventListener("abort", () => source.destroy(), { once: true });
  return Readable.toWeb(source) as ReadableStream;
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const video = await getBallSourceVideo(id);
    const requestedRange = request.headers.get("range");
    if (!requestedRange) {
      return new Response(stream(video.filePath, request), {
        headers: headers(video.filename, video.sha256, video.size),
      });
    }
    const range = parseByteRange(requestedRange, video.size);
    if (!range) {
      return new Response(null, {
        status: 416,
        headers: {
          "Accept-Ranges": "bytes",
          "Content-Range": `bytes */${video.size}`,
          "Cache-Control": "private, no-store",
        },
      });
    }
    const responseHeaders = headers(
      video.filename,
      video.sha256,
      range.end - range.start + 1,
    );
    responseHeaders.set(
      "Content-Range",
      `bytes ${range.start}-${range.end}/${video.size}`,
    );
    return new Response(
      stream(video.filePath, request, range.start, range.end),
      { status: 206, headers: responseHeaders },
    );
  } catch (error) {
    if (error instanceof BallLabelingTaskNotFoundError) {
      return Response.json({ error: "Task not found" }, { status: 404 });
    }
    if (error instanceof BallLabelingVideoValidationError) {
      return Response.json(
        { error: "Source video failed provenance validation" },
        { status: 409, headers: { "Cache-Control": "no-store" } },
      );
    }
    return Response.json(
      { error: "Source video is unavailable" },
      { status: 503, headers: { "Cache-Control": "no-store" } },
    );
  }
}
