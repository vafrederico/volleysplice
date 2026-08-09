import { createReadStream, promises as fs } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import type { NextRequest } from "next/server";
import { getAnalysesRoot } from "@/lib/storage";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ANALYSIS_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/;
const ASSETS: Record<string, string> = {
  "proxy.mp4": "video/mp4",
  "court-preview.jpg": "image/jpeg",
};

type RouteContext = {
  params: Promise<{ analysisId: string; filename: string }>;
};

async function resolveAsset(context: RouteContext) {
  const { analysisId, filename } = await context.params;
  if (!ANALYSIS_ID.test(analysisId) || !(filename in ASSETS)) return null;
  const assetPath = path.join(getAnalysesRoot(), analysisId, filename);
  try {
    const stats = await fs.stat(assetPath);
    return stats.isFile() ? { assetPath, filename, size: stats.size } : null;
  } catch {
    return null;
  }
}

function headers(filename: string, size: number): Headers {
  return new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "private, no-store",
    "Content-Length": String(size),
    "Content-Type": ASSETS[filename],
    "X-Content-Type-Options": "nosniff",
  });
}

function parseRange(value: string, size: number): { start: number; end: number } | null {
  const match = /^bytes=(\d*)-(\d*)$/.exec(value.trim());
  if (!match || (!match[1] && !match[2])) return null;
  let start: number;
  let end: number;
  if (!match[1]) {
    const suffixLength = Number(match[2]);
    if (!Number.isSafeInteger(suffixLength) || suffixLength <= 0) return null;
    start = Math.max(0, size - suffixLength);
    end = size - 1;
  } else {
    start = Number(match[1]);
    end = match[2] ? Number(match[2]) : size - 1;
  }
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || start >= size || end < start) return null;
  return { start, end: Math.min(end, size - 1) };
}

export async function HEAD(_request: NextRequest, context: RouteContext) {
  const asset = await resolveAsset(context);
  if (!asset) return new Response(null, { status: 404 });
  return new Response(null, { status: 200, headers: headers(asset.filename, asset.size) });
}

export async function GET(request: NextRequest, context: RouteContext) {
  const asset = await resolveAsset(context);
  if (!asset) return new Response("Not found", { status: 404 });
  const requestedRange = request.headers.get("range");
  const responseHeaders = headers(asset.filename, asset.size);
  let status = 200;
  let options: { start?: number; end?: number } = {};
  if (requestedRange) {
    const range = parseRange(requestedRange, asset.size);
    if (!range) {
      responseHeaders.set("Content-Length", "0");
      responseHeaders.set("Content-Range", `bytes */${asset.size}`);
      return new Response(null, { status: 416, headers: responseHeaders });
    }
    status = 206;
    options = range;
    responseHeaders.set("Content-Length", String(range.end - range.start + 1));
    responseHeaders.set("Content-Range", `bytes ${range.start}-${range.end}/${asset.size}`);
  }
  const stream = createReadStream(asset.assetPath, options);
  return new Response(Readable.toWeb(stream) as ReadableStream, { status, headers: responseHeaders });
}
