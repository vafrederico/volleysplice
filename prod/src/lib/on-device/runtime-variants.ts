export const ON_DEVICE_RUNTIME_VARIANTS = [
  "linear-v1",
  "libswresample-wasm-v1",
] as const;

export type OnDeviceRuntimeVariant = (typeof ON_DEVICE_RUNTIME_VARIANTS)[number];

// The filtered FFmpeg-compatible resampler is the evaluated production choice.
export const DEFAULT_ON_DEVICE_RUNTIME_VARIANT: OnDeviceRuntimeVariant =
  "libswresample-wasm-v1";
