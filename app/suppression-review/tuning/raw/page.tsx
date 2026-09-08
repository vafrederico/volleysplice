import type { Metadata } from "next";

import reviewData from "@/data/any-overlap-raw-retrained-suppression-review.json";

import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "Raw-connected suppression review · VolleySplice",
  robots: { index: false, follow: false },
};

export default function RawConnectedRetrainedReviewPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
