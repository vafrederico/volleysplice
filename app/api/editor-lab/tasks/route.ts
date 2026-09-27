import { neuralComparisonCatalog } from "@/lib/server/neural-comparison";
import { getPreparedLabelingCatalog } from "@/lib/server/labeling-tasks";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const [entries, catalog] = await Promise.all([neuralComparisonCatalog(), getPreparedLabelingCatalog()]);
    const available = new Set(catalog.tasks.map(task => task.id));
    return Response.json({ tasks: entries.filter(entry => available.has(entry.id)) }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    console.error("Editor lab catalog could not be loaded", error);
    return Response.json({ error: "Editor lab recordings are unavailable." }, { status: 503 });
  }
}
