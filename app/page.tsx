import { ReviewEditor } from "@/components/review-editor";
import { loadReviewCatalog } from "@/lib/server/review-catalog";

export const dynamic = "force-dynamic";

type HomeProps = {
  searchParams: Promise<{
    analysis?: string | string[];
    video?: string | string[];
    time?: string | string[];
  }>;
};

export default async function Home({ searchParams }: HomeProps) {
  const [catalog, requested] = await Promise.all([loadReviewCatalog(), searchParams]);
  const requestedAnalysisId =
    typeof requested.analysis === "string" ? requested.analysis : null;
  const requestedVideoId = typeof requested.video === "string" ? requested.video : null;
  const requestedAnalysis = catalog.analyses.find(
    (candidate) => candidate.id === requestedAnalysisId,
  );
  const selectedVideo =
    catalog.videos.find(
      (video) => video.id === (requestedAnalysis?.recordingId ?? requestedVideoId),
    ) ?? catalog.videos[0] ?? null;
  const analyses = selectedVideo
    ? catalog.analyses.filter((candidate) => candidate.recordingId === selectedVideo.id)
    : [];
  const defaultAnalysis =
    analyses.find(
      (candidate) =>
        candidate.id === `model-full-percentile-v1--${selectedVideo?.id}`,
    ) ??
    analyses.find(
      (candidate) =>
        candidate.kind === "model" && candidate.modelVersion === "full-percentile-v1",
    ) ??
    analyses.find((candidate) => candidate.kind === "model") ??
    analyses.find((candidate) => candidate.kind === "heuristic" && candidate.id.endsWith("-v2")) ??
    analyses.find((candidate) => candidate.kind === "heuristic") ??
    analyses[0] ??
    null;
  const analysis =
    requestedAnalysis?.recordingId === selectedVideo?.id ? requestedAnalysis : defaultAnalysis;
  const requestedTime = typeof requested.time === "string" ? Number(requested.time) : 0;
  const initialTime =
    Number.isFinite(requestedTime) && requestedTime >= 0
      ? Math.min(analysis?.duration ?? 0, requestedTime)
      : 0;

  return (
    <ReviewEditor
      key={analysis?.id ?? "demo"}
      initialAnalysis={analysis}
      analysisOptions={selectedVideo?.analyses ?? []}
      videoOptions={catalog.videos}
      comparisonAnalyses={analyses}
      initialTime={initialTime}
    />
  );
}
