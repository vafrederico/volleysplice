import { createReadStream } from "node:fs";
import { Readable } from "node:stream";

import {
  getPreparedLabelingTask,
  LabelingTaskNotFoundError,
} from "@/lib/server/labeling-tasks";

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

function videoHeaders(filename: string, contentLength: number): Headers {
  return new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, no-store",
    "Content-Disposition": `inline; filename*=UTF-8''${encodeURIComponent(filename)}`,
    "Content-Length": String(contentLength),
    "Content-Type": "video/mp4",
    "X-Content-Type-Options": "nosniff",
  });
}

function streamFile(
  filePath: string,
  request: Request,
  start?: number,
  end?: number,
): ReadableStream {
  const nodeStream = createReadStream(filePath, { start, end });
  request.signal.addEventListener("abort", () => nodeStream.destroy(), { once: true });
  return Readable.toWeb(nodeStream) as ReadableStream;
}

export async function GET(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  try {
    const { id } = await params;
    const task = await getPreparedLabelingTask(id);
    const rangeValue = request.headers.get("range");
    if (!rangeValue) {
      return new Response(streamFile(task.proxyPath, request), {
        status: 200,
        headers: videoHeaders(task.document.recording.videoFilename, task.proxySize),
      });
    }
    const range = parseByteRange(rangeValue, task.proxySize);
    if (!range) {
      return new Response(null, {
        status: 416,
        headers: {
          "Accept-Ranges": "bytes",
          "Content-Range": `bytes */${task.proxySize}`,
        },
      });
    }
    const headers = videoHeaders(
      task.document.recording.videoFilename,
      range.end - range.start + 1,
    );
    headers.set("Content-Range", `bytes ${range.start}-${range.end}/${task.proxySize}`);
    return new Response(
      streamFile(task.proxyPath, request, range.start, range.end),
      { status: 206, headers },
    );
  } catch (error) {
    if (error instanceof LabelingTaskNotFoundError) {
      return Response.json({ error: "Task not found" }, { status: 404 });
    }
    return Response.json({ error: "Video is unavailable" }, { status: 503 });
  }
}
