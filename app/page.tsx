import { ReviewEditor } from "@/components/review-editor";
import { loadLatestAnalysis } from "@/lib/analysis";

export const dynamic = "force-dynamic";

export default async function Home() {
  const analysis = await loadLatestAnalysis();
  return <ReviewEditor initialAnalysis={analysis} />;
}
