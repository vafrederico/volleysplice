import path from "node:path";
import type { Metadata } from "next";

import {
  getServingSideFlightEvaluationPath,
  loadServingSideFlightReview,
} from "@/lib/server/serving-side-flight-review";

import { ServingSideFlightReviewClient } from "./serving-side-flight-review-client";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export const metadata: Metadata = {
  title: "Serving-side flight error review · VolleySplice",
  description:
    "Annotate visibility and timing failure modes for serving-side flight predictions.",
  robots: { index: false, follow: false },
};

type Props = {
  searchParams: Promise<{
    rally?: string | string[];
  }>;
};

export default async function ServingSideFlightReviewPage({
  searchParams,
}: Props) {
  const requested = await searchParams;
  const evaluationPath = getServingSideFlightEvaluationPath();
  try {
    const data = await loadServingSideFlightReview();
    const initialRallyId =
      typeof requested.rally === "string" &&
      data.results.some((result) => result.rallyId === requested.rally)
        ? requested.rally
        : null;
    return (
      <ServingSideFlightReviewClient
        data={data}
        evaluationPath={path.basename(evaluationPath)}
        initialRallyId={initialRallyId}
      />
    );
  } catch (error) {
    return (
      <ServingSideFlightReviewClient
        data={null}
        evaluationPath={evaluationPath}
        initialRallyId={null}
        loadError={error instanceof Error ? error.message : String(error)}
      />
    );
  }
}
