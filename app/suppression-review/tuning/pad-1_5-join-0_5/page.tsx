import type { Metadata } from "next";

import reviewData from "@/data/any-overlap-retrained-pad-1_5-join-0_5.json";

import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "1.5s / 0.5s suppression review · VolleySplice",
  robots: { index: false, follow: false },
};

export default function AgreementPadOnePointFiveJoinHalfPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
