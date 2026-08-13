import type { Metadata } from "next";

import { OnDeviceClient } from "./on-device-client";

export const metadata: Metadata = {
  title: "On-device cut · VolleyCut",
  description: "Analyze and export volleyball footage locally in a desktop browser.",
};

export default function OnDevicePage() {
  return <OnDeviceClient />;
}
