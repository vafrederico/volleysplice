import type { Metadata } from "next";

import reviewData from "@/data/any-overlap-retrained-pad-2-join-0_5.json";

import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "2s / 0.5s suppression review · VolleyCut",
  robots: { index: false, follow: false },
};

export default function AgreementPadTwoJoinHalfPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
