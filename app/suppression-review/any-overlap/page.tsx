import type { Metadata } from "next";

import reviewData from "@/data/any-overlap-suppression-review.json";

import { SuppressionReviewClient } from "../suppression-review-client";
import type { SuppressionReviewDataset } from "../types";

export const metadata: Metadata = {
  title: "Padded/joined any-overlap suppression review · VolleyCut",
  description:
    "Inspect missed rallies when cross-model support in a padded and joined export component protects its full production span.",
  robots: { index: false, follow: false },
};

export default function AnyOverlapSuppressionReviewPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
