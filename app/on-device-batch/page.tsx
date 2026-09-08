import type { Metadata } from "next";

import { parseOnDeviceRuntimeVariantQueryValue } from "@/lib/on-device/runtime-variants";

import { OnDeviceBatchClient } from "./on-device-batch-client";

export const metadata: Metadata = {
  title: "On-device batch runner · VolleySplice",
  description: "Generate browser-native model predictions for the comparison dataset.",
  robots: { index: false, follow: false },
};

export default async function OnDeviceBatchPage({
  searchParams,
}: {
  searchParams: Promise<{ variant?: string | string[] }>;
}) {
  const requested = (await searchParams).variant;
  return (
    <OnDeviceBatchClient
      runtimeVariant={parseOnDeviceRuntimeVariantQueryValue(requested)}
    />
  );
}
