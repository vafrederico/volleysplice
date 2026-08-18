import type { Metadata } from "next";
import reviewData from "@/data/corrected-v3-suppression-zero-miss.json";
import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "Held decoder · zero-non-exempt-miss review · VolleyCut",
  robots: { index: false, follow: false },
};

export default function CorrectedV3ZeroMissReviewPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
