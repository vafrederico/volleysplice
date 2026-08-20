import path from "node:path";
import type { Metadata } from "next";

import {
  getServingSideResultsEvaluationPath,
  loadServingSideResults,
} from "@/lib/server/serving-side-results";

import { ServingSideResultsClient } from "./serving-side-results-client";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const metadata: Metadata = {
  title: "Serving-side model results · VolleyCut",
  description:
    "Compare frozen serving-side model predictions with human labels.",
  robots: { index: false, follow: false },
};

type Props = {
  searchParams: Promise<{
    video?: string | string[];
    outcome?: string | string[];
  }>;
};

export default async function ServingSideResultsPage({ searchParams }: Props) {
  const evaluationPath = getServingSideResultsEvaluationPath();
  const requested = await searchParams;
  try {
    const data = await loadServingSideResults();
    const requestedRecordingId =
      typeof requested.video === "string" &&
      data.recordings.some(
        (recording) => recording.recordingId === requested.video,
      )
        ? requested.video
        : null;
    const requestedOutcome =
      requested.outcome === "all" ||
      requested.outcome === "correct" ||
      requested.outcome === "wrong" ||
      requested.outcome === "near-as-far" ||
      requested.outcome === "far-as-near"
        ? requested.outcome
        : "wrong";
    return (
      <ServingSideResultsClient
        data={data}
        evaluationPath={path.basename(evaluationPath)}
        initialRecordingId={requestedRecordingId}
        initialOutcome={requestedOutcome}
      />
    );
  } catch (error) {
    return (
      <ServingSideResultsClient
        data={null}
        evaluationPath={evaluationPath}
        initialRecordingId={null}
        initialOutcome="wrong"
        loadError={error instanceof Error ? error.message : String(error)}
      />
    );
  }
}
