import type { Metadata } from "next";
import reviewData from "@/data/corrected-v3-suppression-raw-connected.json";
import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "Held decoder · raw-connected review · VolleySplice",
  robots: { index: false, follow: false },
};

export default function CorrectedV3RawReviewPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
