import { ReviewEditor } from "@/components/review-editor";
import { loadAnalyses } from "@/lib/analysis";
import type { AnalysisOption } from "@/lib/analysis-types";

export const dynamic = "force-dynamic";

type HomeProps = {
  searchParams: Promise<{ analysis?: string | string[] }>;
};

export default async function Home({ searchParams }: HomeProps) {
  const analyses = await loadAnalyses();
  const requested = (await searchParams).analysis;
  const requestedId = typeof requested === "string" ? requested : null;
  const analysis = analyses.find((candidate) => candidate.id === requestedId) ?? analyses[0] ?? null;
  const analysisOptions: AnalysisOption[] = analyses.map((candidate) => ({
    id: candidate.id,
    title: candidate.title,
    duration: candidate.duration,
    rallyCount: candidate.rallies.length,
  }));

  return (
    <ReviewEditor
      key={analysis?.id ?? "demo"}
      initialAnalysis={analysis}
      analysisOptions={analysisOptions}
    />
  );
}
