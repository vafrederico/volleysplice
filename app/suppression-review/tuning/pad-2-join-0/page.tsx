import type { Metadata } from "next";

import reviewData from "@/data/any-overlap-retrained-pad-2-join-0.json";

import { SuppressionReviewClient } from "../../suppression-review-client";
import type { SuppressionReviewDataset } from "../../types";

export const metadata: Metadata = {
  title: "2s / no-join suppression review · VolleySplice",
  robots: { index: false, follow: false },
};

export default function AgreementPadTwoNoJoinPage() {
  return (
    <SuppressionReviewClient dataset={reviewData as SuppressionReviewDataset} />
  );
}
