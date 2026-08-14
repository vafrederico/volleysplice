export const ON_DEVICE_RUNTIME_VARIANTS = [
  "linear-v1",
  "libswresample-wasm-v1",
] as const;

export type OnDeviceRuntimeVariant = (typeof ON_DEVICE_RUNTIME_VARIANTS)[number];

export const DEFAULT_ON_DEVICE_RUNTIME_VARIANT: OnDeviceRuntimeVariant = "linear-v1";

const runtimeVariants = new Set<string>(ON_DEVICE_RUNTIME_VARIANTS);

export function isOnDeviceRuntimeVariant(value: unknown): value is OnDeviceRuntimeVariant {
  return typeof value === "string" && runtimeVariants.has(value);
}

export function parseOnDeviceRuntimeVariantQueryValue(
  value: string | string[] | undefined,
): OnDeviceRuntimeVariant | null {
  if (value === undefined) return DEFAULT_ON_DEVICE_RUNTIME_VARIANT;
  return isOnDeviceRuntimeVariant(value) ? value : null;
}
