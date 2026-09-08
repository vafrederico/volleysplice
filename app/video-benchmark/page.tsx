import type { Metadata } from "next";

import { VideoBenchmarkClient } from "./video-benchmark-client";

export const metadata: Metadata = {
  title: "Video performance lab · VolleySplice",
  description:
    "Benchmark browser-native video decoding and visual feature generation.",
  robots: { index: false, follow: false },
};

export default function VideoBenchmarkPage() {
  return <VideoBenchmarkClient />;
}
