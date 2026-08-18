import type { Metadata } from "next";

import reviewData from "@/data/single-model-suppression-review.json";

import { SuppressionReviewClient } from "./suppression-review-client";
import type { SuppressionReviewDataset } from "./types";

export const metadata: Metadata = {
  title: "Pointwise-overlap suppression review · VolleyCut",
  description:
    "Visually inspect rallies affected by suppression of one-model-only production output.",
  robots: { index: false, follow: false },
};

export default function SuppressionReviewPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
