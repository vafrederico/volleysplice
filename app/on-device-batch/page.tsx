import type { Metadata } from "next";

import { OnDeviceBatchClient } from "./on-device-batch-client";

export const metadata: Metadata = {
  title: "On-device batch runner · VolleyCut",
  description: "Generate browser-native model predictions for the comparison dataset.",
  robots: { index: false, follow: false },
};

export default function OnDeviceBatchPage() {
  return <OnDeviceBatchClient />;
}
