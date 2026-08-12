import type { Metadata } from "next";

import { BallReviewBenchmark } from "@/components/ball-review-benchmark";
import { getBallReviewBenchmarkBundle } from "@/lib/server/ball-review-benchmark";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Ball review effort benchmark · VolleyCut",
  description: "Visual comparison of nine blinded ball-presence review runs.",
};

export default async function BallReviewBenchmarkPage() {
  const benchmark = await getBallReviewBenchmarkBundle();
  return <BallReviewBenchmark benchmark={benchmark} />;
}
