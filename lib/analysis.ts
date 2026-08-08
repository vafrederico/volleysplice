import { promises as fs } from "node:fs";
import path from "node:path";
import type { CourtLine, ReviewAnalysis } from "@/lib/analysis-types";
import type { Rally } from "@/lib/edit-list";

const ANALYSIS_ID = /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/;
const analysesRoot = path.join(process.cwd(), "data", "analyses");

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

export function parseAnalysis(value: unknown): ReviewAnalysis | null {
  const root = record(value);
  if (!root || root.schemaVersion !== 1 || typeof root.id !== "string" || !ANALYSIS_ID.test(root.id)) return null;
  const proxy = record(root.proxy);
  const source = record(root.source);
  const analysis = record(root.analysis);
  const court = record(analysis?.court);
  const duration = finiteNumber(proxy?.duration);
  if (!proxy || !duration || duration <= 0 || !source || !analysis || !court) return null;

  const seenIds = new Set<string>();
  const rallies: Rally[] = [];
  if (Array.isArray(root.rallies)) {
    for (const candidate of root.rallies) {
      const item = record(candidate);
      if (!item || typeof item.id !== "string" || seenIds.has(item.id)) continue;
      const rawStart = finiteNumber(item.start);
      const rawEnd = finiteNumber(item.end);
      const rawConfidence = finiteNumber(item.confidence);
      if (rawStart === null || rawEnd === null || rawConfidence === null) continue;
      const start = Math.max(0, Math.min(duration, rawStart));
      const end = Math.max(0, Math.min(duration, rawEnd));
      if (end <= start) continue;
      seenIds.add(item.id);
      rallies.push({
        id: item.id,
        start,
        end,
        confidence: Math.max(0, Math.min(1, rawConfidence)),
        included: item.included !== false,
      });
    }
  }
  rallies.sort((a, b) => a.start - b.start || a.end - b.end);

  const lines: CourtLine[] = [];
  if (Array.isArray(court.lines)) {
    for (const candidate of court.lines) {
      const line = record(candidate);
      const coordinates = [line?.x1, line?.y1, line?.x2, line?.y2].map(finiteNumber);
      if (coordinates.some((coordinate) => coordinate === null)) continue;
      const [x1, y1, x2, y2] = coordinates as number[];
      lines.push({
        x1: Math.max(0, Math.min(1, x1)),
        y1: Math.max(0, Math.min(1, y1)),
        x2: Math.max(0, Math.min(1, x2)),
        y2: Math.max(0, Math.min(1, y2)),
      });
    }
  }

  return {
    id: root.id,
    title: typeof root.title === "string" && root.title.trim() ? root.title : root.id,
    duration,
    width: Math.max(1, finiteNumber(proxy.width) ?? 16),
    height: Math.max(1, finiteNumber(proxy.height) ?? 9),
    sourceFilename: typeof source.filename === "string" ? source.filename : "Local video",
    videoUrl: `/api/media/${root.id}/proxy.mp4`,
    courtPreviewUrl: `/api/media/${root.id}/court-preview.jpg`,
    courtConfidence: Math.max(0, Math.min(1, finiteNumber(court.confidence) ?? 0)),
    courtSource: typeof court.source === "string" ? court.source : "unknown",
    courtLines: lines,
    cameraStability: Math.max(0, Math.min(1, finiteNumber(analysis.cameraStability) ?? 0)),
    warnings: Array.isArray(analysis.warnings)
      ? analysis.warnings.filter((warning): warning is string => typeof warning === "string")
      : [],
    rallies,
  };
}

export async function loadLatestAnalysis(): Promise<ReviewAnalysis | null> {
  let entries;
  try {
    entries = await fs.readdir(analysesRoot, { withFileTypes: true });
  } catch {
    return null;
  }
  const candidates = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory() && ANALYSIS_ID.test(entry.name))
      .map(async (entry) => {
        const filename = path.join(analysesRoot, entry.name, "analysis.json");
        try {
          const stats = await fs.stat(filename);
          return { filename, modified: stats.mtimeMs };
        } catch {
          return null;
        }
      }),
  );
  candidates.sort((a, b) => (b?.modified ?? 0) - (a?.modified ?? 0));
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      const parsed = parseAnalysis(JSON.parse(await fs.readFile(candidate.filename, "utf-8")));
      if (parsed) return parsed;
    } catch {
      // Ignore incomplete or malformed local runs and try the next one.
    }
  }
  return null;
}
