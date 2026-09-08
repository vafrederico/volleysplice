import type { Metadata } from "next";

import { AudioBenchmarkClient } from "./audio-benchmark-client";

export const metadata: Metadata = {
  title: "Audio performance lab · VolleySplice",
  description:
    "Benchmark browser-native audio feature generation by pipeline phase.",
  robots: { index: false, follow: false },
};

export default function AudioBenchmarkPage() {
  return <AudioBenchmarkClient />;
}
