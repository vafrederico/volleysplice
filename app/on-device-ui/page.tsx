import type { Metadata } from "next";

import { OnDeviceClient, type OnDeviceUiFixture } from "@/app/on-device/on-device-client";
import browserPrediction from "@/public/on-device/browser-prediction-y9.json";

export const metadata: Metadata = {
  title: "On-device UI fixture · VolleyCut",
  description: "Review a cached browser on-device prediction without decoding video.",
};

export default function OnDeviceUiPage() {
  return <OnDeviceClient fixture={browserPrediction as OnDeviceUiFixture} />;
}
