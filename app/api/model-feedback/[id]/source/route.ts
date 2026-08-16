import { createReadStream } from "node:fs";
import { lstat } from "node:fs/promises";
import { Readable } from "node:stream";

import {
  getModelFeedbackImport,
  ModelFeedbackImportNotFoundError,
} from "../../../../../lib/server/model-feedback-store.ts";

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

function responseHeaders(
  filename: string,
  contentType: string,
  contentLength: number,
): Headers {
  const extension = filename.toLowerCase().split(".").pop();
  const inferredType: Record<string, string> = {
    mkv: "video/x-matroska",
    mov: "video/quicktime",
    mp4: "video/mp4",
    webm: "video/webm",
  };
  return new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, no-store",
    "Content-Disposition": `inline; filename*=UTF-8''${encodeURIComponent(filename)}`,
    "Content-Length": String(contentLength),
    "Content-Type": contentType.startsWith("video/")
      ? contentType
      : (inferredType[extension ?? ""] ?? "application/octet-stream"),
    "X-Content-Type-Options": "nosniff",
  });
}

function streamSource(
  filePath: string,
  request: Request,
  start?: number,
  end?: number,
): ReadableStream {
  const source = createReadStream(filePath, { start, end });
  request.signal.addEventListener("abort", () => source.destroy(), {
    once: true,
  });
  return Readable.toWeb(source) as ReadableStream;
}

async function serve(
  request: Request,
  { params }: { params: Promise<{ id: string }> },
  head: boolean,
): Promise<Response> {
  try {
    const { id } = await params;
    const stored = await getModelFeedbackImport(id);
    if (
      stored.sourcePath === null ||
      stored.sourceSize === null ||
      stored.sourceMimeType === null
    ) {
      return Response.json(
        { error: "This feedback import has no linked source" },
        { status: 404 },
      );
    }
    const source = await lstat(stored.sourcePath);
    if (!source.isFile() || source.size !== stored.sourceSize) {
      return Response.json(
        { error: "The linked source has changed" },
        { status: 409 },
      );
    }
    const rangeValue = request.headers.get("range");
    if (!rangeValue) {
      return new Response(
        head ? null : streamSource(stored.sourcePath, request),
        {
          status: 200,
          headers: responseHeaders(
            stored.sourceName,
            stored.sourceMimeType,
            stored.sourceSize,
          ),
        },
      );
    }
    const range = parseByteRange(rangeValue, stored.sourceSize);
    if (!range) {
      return new Response(null, {
        status: 416,
        headers: {
          "Accept-Ranges": "bytes",
          "Content-Range": `bytes */${stored.sourceSize}`,
        },
      });
    }
    const headers = responseHeaders(
      stored.sourceName,
      stored.sourceMimeType,
      range.end - range.start + 1,
    );
    headers.set(
      "Content-Range",
      `bytes ${range.start}-${range.end}/${stored.sourceSize}`,
    );
    return new Response(
      head
        ? null
        : streamSource(stored.sourcePath, request, range.start, range.end),
      { status: 206, headers },
    );
  } catch (error) {
    if (error instanceof ModelFeedbackImportNotFoundError) {
      return Response.json({ error: "Import not found" }, { status: 404 });
    }
    return Response.json(
      { error: "The linked source is unavailable" },
      { status: 503 },
    );
  }
}

export async function HEAD(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  return serve(request, context, true);
}

export async function GET(
  request: Request,
  context: { params: Promise<{ id: string }> },
) {
  return serve(request, context, false);
}
