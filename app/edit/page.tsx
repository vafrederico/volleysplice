import Link from "next/link";

import { CutEditor } from "@/components/cut-editor";
import type { AnalysisOption } from "@/lib/analysis-types";
import { PRODUCTION_ENSEMBLE_MODEL_ID } from "@/lib/production-model";
import { loadReviewCatalog } from "@/lib/server/review-catalog";

export const dynamic = "force-dynamic";

type EditPageProps = {
  searchParams: Promise<{
    analysis?: string | string[];
    video?: string | string[];
  }>;
};

function preferredAnalysis(analyses: AnalysisOption[]): AnalysisOption | null {
  return (
    analyses.find((analysis) =>
      analysis.id === `${PRODUCTION_ENSEMBLE_MODEL_ID}--${analysis.recordingId}`
    ) ??
    analyses.find((analysis) => analysis.kind === "model") ??
    analyses.find((analysis) => analysis.kind === "sol") ??
    analyses.find((analysis) => analysis.kind === "gold") ??
    analyses[0] ??
    null
  );
}
export default async function EditPage({ searchParams }: EditPageProps) {
  const [catalog, requested] = await Promise.all([loadReviewCatalog(), searchParams]);
  const requestedAnalysisId =
    typeof requested.analysis === "string" ? requested.analysis : null;
  const requestedVideoId = typeof requested.video === "string" ? requested.video : null;
  const requestedAnalysis = catalog.analyses.find(
    (analysis) => analysis.id === requestedAnalysisId,
  );
  const video =
    catalog.videos.find(
      (candidate) => candidate.id === (requestedAnalysis?.recordingId ?? requestedVideoId),
    ) ?? catalog.videos[0] ?? null;
  const analysisOptions = video?.analyses ?? [];
  const defaultAnalysis = preferredAnalysis(analysisOptions);
  const analysis = catalog.analyses.find(
    (candidate) =>
      candidate.recordingId === video?.id &&
      candidate.id === (requestedAnalysis?.id ?? defaultAnalysis?.id),
  );

  if (!video || !analysis) {
    return (
      <main style={{ padding: "min(8vw, 5rem)" }}>
        <p>No cached video labels are available for editing.</p>
        <Link href="/">Return to VolleyCut Lab</Link>
      </main>
    );
  }

  return (
    <CutEditor
      key={analysis.id}
      initialAnalysis={analysis}
      analysisOptions={analysisOptions}
      videoOptions={catalog.videos}
    />
  );
}
