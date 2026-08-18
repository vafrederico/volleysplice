import type { Metadata } from "next";
import reviewData from "@/data/corrected-v3-suppression-aggressive.json";
import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "Held decoder · aggressive suppression review · VolleyCut",
  robots: { index: false, follow: false },
};

export default function CorrectedV3AggressiveReviewPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
