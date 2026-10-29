import { ANALYSIS_FPS, FRAME_FEATURE_NAMES } from "./feature-schema.ts";
import type {
  FeatureReductionKernel,
  NormalizedRoi,
  OnDeviceMediaInfo,
  VideoDecoderAcceleration,
  VideoDecodeStrategy,
} from "./types.ts";

export type LocalFeatureSource = {
  name: string;
  size: number;
  lastModified: number;
};

export function visualFeatureCacheKey(
  source: LocalFeatureSource,
  info: OnDeviceMediaInfo,
  roi: NormalizedRoi,
  experiment?: {
    decodeStrategy: VideoDecodeStrategy;
    decoderAcceleration: VideoDecoderAcceleration;
    reductionKernel?: FeatureReductionKernel;
    areaNearest?: boolean;
  },
): string {
  let featureSignature = 0x811c9dc5;
  for (const character of FRAME_FEATURE_NAMES.join("|")) {
    featureSignature ^= character.charCodeAt(0);
    featureSignature = Math.imul(featureSignature, 0x01000193) >>> 0;
  }
  return JSON.stringify({
    schema: 1,
    featureSchema: featureSignature.toString(16),
    ...(experiment?.areaNearest ? { visualExtractor: "opencv-area-nearest-grid-v1" } : {}),
    analysisFps: ANALYSIS_FPS,
    source: [source.name, source.size, source.lastModified],
    media: [info.duration, info.width, info.height, info.rotation, info.videoCodecString],
    roi: [roi.x, roi.y, roi.width, roi.height],
    ...(experiment ? { experiment } : {}),
  });
}
