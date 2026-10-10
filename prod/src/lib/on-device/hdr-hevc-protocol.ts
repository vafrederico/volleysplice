export type HdrDecodedFrame = {
  pixels: ArrayBuffer;
  width: number;
  height: number;
  timestamp: number;
  duration: number;
  primaries: number;
  transfer: number;
  matrix: number;
  range: number;
};
export type HdrDecoderRequest =
  | { kind: "open"; url: string; extra: Uint8Array }
  | { kind: "reset" }
  | { kind: "send"; data: Uint8Array | null; timestamp: number; duration: number }
  | { kind: "receive" };
export type HdrDecoderResponse = { frame?: HdrDecodedFrame | null; error?: string };
