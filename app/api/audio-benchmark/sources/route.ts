import { readdir, stat } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const DEFAULT_AUDIO_BENCHMARK_ROOT = "/mnt/freenas/volleycut-raw-no-backup";

function unavailable() {
  return Response.json(
    { error: "NAS benchmark sources are available only in development." },
    { status: 404, headers: { "Cache-Control": "private, no-store" } },
  );
}

export async function GET() {
  if (process.env.NODE_ENV !== "development") return unavailable();
  const root = path.resolve(
    process.env.VOLLEYCUT_AUDIO_BENCHMARK_ROOT ?? DEFAULT_AUDIO_BENCHMARK_ROOT,
  );
  try {
    const entries = (await readdir(root, { withFileTypes: true }))
      .filter(
        (entry) => entry.isFile() && entry.name.toLowerCase().endsWith(".mp4"),
      )
      .sort((left, right) => left.name.localeCompare(right.name));
    const sources = await Promise.all(
      entries.map(async (entry) => {
        const metadata = await stat(path.join(root, entry.name));
        return {
          name: entry.name,
          size: metadata.size,
          modifiedAt: metadata.mtime.toISOString(),
          mediaUrl: `/api/audio-benchmark/media/${encodeURIComponent(entry.name)}`,
        };
      }),
    );
    return Response.json(
      { root, sources },
      { headers: { "Cache-Control": "private, no-store" } },
    );
  } catch {
    return Response.json(
      { error: "The audio benchmark source directory is unavailable." },
      { status: 503, headers: { "Cache-Control": "private, no-store" } },
    );
  }
}
