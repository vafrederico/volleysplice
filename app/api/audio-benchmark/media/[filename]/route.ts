import { createReadStream } from "node:fs";
import { lstat, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DEFAULT_AUDIO_BENCHMARK_ROOT = "/mnt/freenas/volleycut-raw-no-backup";

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

function mediaHeaders(filename: string, contentLength: number): Headers {
  return new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, no-store",
    "Content-Disposition": `inline; filename*=UTF-8''${encodeURIComponent(filename)}`,
    "Content-Length": String(contentLength),
    "Content-Type": "video/mp4",
    "X-Content-Type-Options": "nosniff",
  });
}

async function resolveSource(filename: string) {
  if (process.env.NODE_ENV !== "development") return null;
  if (
    filename !== path.basename(filename) ||
    !/^[a-z0-9_. -]+\.mp4$/i.test(filename)
  )
    return null;
  const root = path.resolve(
    process.env.VOLLEYCUT_AUDIO_BENCHMARK_ROOT ?? DEFAULT_AUDIO_BENCHMARK_ROOT,
  );
  const filePath = path.join(root, filename);
  const [linkMetadata, metadata] = await Promise.all([
    lstat(filePath),
    stat(filePath),
  ]);
  if (linkMetadata.isSymbolicLink() || !metadata.isFile()) return null;
  return { filePath, size: metadata.size };
}

async function serve(
  request: Request,
  params: Promise<{ filename: string }>,
  head: boolean,
) {
  try {
    const { filename } = await params;
    const source = await resolveSource(filename);
    if (!source) {
      return Response.json(
        { error: "Benchmark source not found" },
        { status: 404 },
      );
    }
    const rangeValue = request.headers.get("range");
    if (!rangeValue) {
      const headers = mediaHeaders(filename, source.size);
      if (head) return new Response(null, { status: 200, headers });
      const nodeStream = createReadStream(
        /* turbopackIgnore: true */ source.filePath,
      );
      request.signal.addEventListener("abort", () => nodeStream.destroy(), {
        once: true,
      });
      return new Response(Readable.toWeb(nodeStream) as ReadableStream, {
        status: 200,
        headers,
      });
    }
    const range = parseByteRange(rangeValue, source.size);
    if (!range) {
      return new Response(null, {
        status: 416,
        headers: {
          "Accept-Ranges": "bytes",
          "Content-Range": `bytes */${source.size}`,
        },
      });
    }
    const headers = mediaHeaders(filename, range.end - range.start + 1);
    headers.set(
      "Content-Range",
      `bytes ${range.start}-${range.end}/${source.size}`,
    );
    if (head) return new Response(null, { status: 206, headers });
    const nodeStream = createReadStream(
      /* turbopackIgnore: true */ source.filePath,
      range,
    );
    request.signal.addEventListener("abort", () => nodeStream.destroy(), {
      once: true,
    });
    return new Response(Readable.toWeb(nodeStream) as ReadableStream, {
      status: 206,
      headers,
    });
  } catch {
    return Response.json(
      { error: "Benchmark source is unavailable" },
      { status: 404 },
    );
  }
}

export function GET(
  request: Request,
  { params }: { params: Promise<{ filename: string }> },
) {
  return serve(request, params, false);
}

export function HEAD(
  request: Request,
  { params }: { params: Promise<{ filename: string }> },
) {
  return serve(request, params, true);
}
