import { ReviewEditor } from "@/components/review-editor";
import type {
  AnalysisOption,
  ReviewAnalysis,
  TrainingCorpusView,
} from "@/lib/analysis-types";
import { PREFERRED_REVIEW_MODEL_ID } from "@/lib/experiment-models";
import { loadReviewCatalog } from "@/lib/server/review-catalog";

export const dynamic = "force-dynamic";

type HomeProps = {
  searchParams: Promise<{
    analysis?: string | string[];
    video?: string | string[];
    time?: string | string[];
    corpus?: string | string[];
  }>;
};

function visibleInCorpus(
  analysis: Pick<ReviewAnalysis | AnalysisOption, "trainingCorpus">,
  corpus: TrainingCorpusView,
): boolean {
  return (
    analysis.trainingCorpus === "reference" ||
    analysis.trainingCorpus === "mixed" ||
    corpus === "both" ||
    analysis.trainingCorpus === corpus
  );
}

export default async function Home({ searchParams }: HomeProps) {
  const [catalog, requested] = await Promise.all([loadReviewCatalog(), searchParams]);
  const requestedAnalysisId =
    typeof requested.analysis === "string" ? requested.analysis : null;
  const requestedVideoId = typeof requested.video === "string" ? requested.video : null;
  const requestedAnalysis = catalog.analyses.find(
    (candidate) => candidate.id === requestedAnalysisId,
  );
  const requestedCorpus =
    requested.corpus === "original" ||
    requested.corpus === "without-beach" ||
    requested.corpus === "both"
      ? requested.corpus
      : null;
  const corpusView: TrainingCorpusView =
    requestedCorpus ??
    (requestedAnalysis?.trainingCorpus === "without-beach"
      ? "without-beach"
      : requestedAnalysis?.trainingCorpus === "original"
        ? "original"
        : "without-beach");
  const selectedVideo =
    catalog.videos.find(
      (video) => video.id === (requestedAnalysis?.recordingId ?? requestedVideoId),
    ) ?? catalog.videos[0] ?? null;
  const analyses = selectedVideo
    ? catalog.analyses.filter(
        (candidate) =>
          candidate.recordingId === selectedVideo.id &&
          visibleInCorpus(candidate, corpusView),
      )
    : [];
  const defaultAnalysis =
    analyses.find(
      (candidate) =>
        candidate.id ===
        `${PREFERRED_REVIEW_MODEL_ID}--${selectedVideo?.id}`,
    ) ??
    analyses.find((candidate) => candidate.kind === "gold") ??
    analyses.find((candidate) => candidate.kind === "sol") ??
    analyses.find(
      (candidate) =>
        corpusView === "without-beach" &&
        candidate.id === `model-nb-audiovisual-v2-final--${selectedVideo?.id}`,
    ) ??
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
    analyses.find((candidate) => candidate.id === requestedAnalysis?.id) ??
    defaultAnalysis;
  const analysisOptions =
    selectedVideo?.analyses.filter((candidate) =>
      visibleInCorpus(candidate, corpusView),
    ) ?? [];
  const requestedTime = typeof requested.time === "string" ? Number(requested.time) : 0;
  const initialTime =
    Number.isFinite(requestedTime) && requestedTime >= 0
      ? Math.min(analysis?.duration ?? 0, requestedTime)
      : 0;

  return (
    <ReviewEditor
      key={analysis?.id ?? "demo"}
      initialAnalysis={analysis}
      analysisOptions={analysisOptions}
      videoOptions={catalog.videos}
      comparisonAnalyses={analyses}
      initialTime={initialTime}
      corpusView={corpusView}
    />
  );
}
