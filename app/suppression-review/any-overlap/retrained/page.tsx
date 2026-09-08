import type { Metadata } from "next";

import reviewData from "@/data/any-overlap-retrained-suppression-review.json";

import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "Retrained padded/joined any-overlap review · VolleySplice",
  description:
    "Inspect missed rallies under padded and joined export-component protection using the overlap-exclusion retrained suppression specialist.",
  robots: { index: false, follow: false },
};

export default function RetrainedAnyOverlapSuppressionReviewPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
