import { EncodedPacketSink, VideoSample, VideoSampleSink, type InputVideoTrack, type VideoEncodingConfig } from "mediabunny";
import { runtimeAssetUrl } from "../runtime-assets.ts";
import { compositeHdrScore, type HdrMask, type HdrTransfer } from "./hdr-color.ts";
import type { HdrDecodedFrame, HdrDecoderRequest, HdrDecoderResponse } from "./hdr-hevc-protocol";

type ColorDescription = { primaries?: string | null; transfer?: string | null; matrix?: string | null; fullRange?: boolean | null };
// TypeScript's DOM enum still omits HDR values implemented by current WebCodecs.
export type HdrColor = { primaries: "bt2020"; matrix: "bt2020-ncl"; transfer: HdrTransfer; fullRange: boolean };
const webColor = (color: HdrColor) => color as unknown as VideoColorSpaceInit;
export function hdrColor(color: ColorDescription): HdrColor | null {
  if (color.transfer !== "hlg" && color.transfer !== "pq") return null;
  if (color.primaries !== "bt2020" || color.matrix !== "bt2020-ncl" || color.fullRange == null) {
    throw new Error("HDR export requires a declared BT.2020 non-constant-luminance color space and range.");
  }
  return { primaries: "bt2020", matrix: "bt2020-ncl", transfer: color.transfer, fullRange: color.fullRange };
}

function assertColor(actual: ColorDescription, expected: HdrColor): void {
  for (const key of ["primaries", "transfer", "matrix", "fullRange"] as const) {
    if (actual[key] !== expected[key]) throw new Error("The video codec changed the HDR color space. Export stopped to avoid incorrect colors.");
  }
}

export function assertHdrEncoding(config: VideoDecoderConfig, color: HdrColor): void {
  if (!/^(?:hvc1|hev1)\.2\./.test(config.codec) && !/^vp09\.02\.\d+\.10(?:\.|$)/.test(config.codec)) {
    throw new Error("The browser did not produce a 10-bit HDR video stream.");
  }
  if (!config.colorSpace) throw new Error("The browser omitted the HDR output color space.");
  assertColor(config.colorSpace, color);
}

/** Capability queries alone may accept a codec that fails or downconverts at encode time. */
export async function chooseHdrEncoder(width: number, height: number, color: HdrColor, source?: VideoSample | null): Promise<{
  encoding: Pick<VideoEncodingConfig, "codec" | "fullCodecString" | "hardwareAcceleration">;
  direct: boolean;
}> {
  const candidates = [
    { codec: "hevc" as const, fullCodecString: "hvc1.2.4.L153.B0", hardwareAcceleration: "prefer-hardware" as const },
    { codec: "vp9" as const, fullCodecString: "vp09.02.51.10", hardwareAcceleration: "prefer-software" as const },
  ];
  for (const candidate of candidates) {
    // Opaque GPU frames can sometimes go straight to the encoder even though
    // CPU readback is unavailable. Preserve that path for score-free exports.
    for (const direct of source ? [true, false] : [false]) {
    const config: VideoEncoderConfig = {
      codec: candidate.fullCodecString, width, height, bitrate: Math.max(2_000_000, width * height * 8),
      framerate: 60, latencyMode: "quality", hardwareAcceleration: candidate.hardwareAcceleration,
    };
    let encoder: VideoEncoder | undefined;
    let frame: VideoFrame | undefined;
    try {
      if (!(await VideoEncoder.isConfigSupported(config)).supported) continue;
      let encoded = false;
      let failure: unknown;
      encoder = new VideoEncoder({
        output: (_chunk, metadata) => {
          try {
            if (!metadata?.decoderConfig) throw new Error("Missing HDR codec configuration.");
            assertHdrEncoding(metadata.decoderConfig, color);
            encoded = true;
          } catch (error) { failure = error; }
        },
        error: error => { failure = error; },
      });
      encoder.configure(config);
      if (direct && source) {
        assertColor(source.colorSpace, color);
        frame = source.toVideoFrame();
      } else {
        const pixels = new Uint16Array(width * height + 2 * Math.ceil(width / 2) * Math.ceil(height / 2));
        pixels.fill(color.fullRange ? 512 : 502, 0, width * height);
        pixels.fill(512, width * height);
        frame = new VideoFrame(pixels, { format: "I420P10" as VideoPixelFormat, codedWidth: width, codedHeight: height, timestamp: 0, colorSpace: webColor(color) });
      }
      encoder.encode(frame, { keyFrame: true });
      await encoder.flush();
      if (failure) throw failure;
      if (encoded) return { encoding: candidate, direct };
    } catch { /* Try the next real 10-bit encoder; never fall back to SDR. */ }
    finally { frame?.close(); if (encoder && encoder.state !== "closed") encoder.close(); }
    }
  }
  throw new Error("This browser has no working 10-bit HDR encoder. Try another desktop browser; HDR export will not silently convert your video to SDR.");
}

class HevcDecoder {
  private worker = new Worker(new URL("./hdr-hevc.worker.ts", import.meta.url), { type: "module" });
  private pending: { resolve: (response: HdrDecoderResponse) => void; reject: (error: Error) => void } | null = null;
  constructor() {
    this.worker.onmessage = (event: MessageEvent<HdrDecoderResponse>) => {
      const pending = this.pending;
      this.pending = null;
      if (event.data.error) pending?.reject(new Error(event.data.error));
      else pending?.resolve(event.data);
    };
    this.worker.onerror = event => {
      this.pending?.reject(new Error(event.message || "The HDR decoder worker failed."));
      this.pending = null;
    };
  }
  request(request: HdrDecoderRequest): Promise<HdrDecoderResponse> {
    if (this.pending) throw new Error("Concurrent HDR decoder request.");
    return new Promise((resolve, reject) => {
      this.pending = { resolve, reject };
      this.worker.postMessage(request);
    });
  }
  close(): void {
    this.worker.terminate();
    this.pending?.reject(new Error("HDR decoder closed."));
    this.pending = null;
  }
}

/** FFmpeg only reconstructs the HEVC base layer: no tone mapping or RGB conversion. */
async function* softwareSamples(track: InputVideoTrack, color: HdrColor, start: number, end: number): AsyncGenerator<VideoSample> {
  const config = await track.getDecoderConfig();
  if (!config?.description) throw new Error("HDR HEVC decoder configuration is missing.");
  const description = config.description;
  const extra = ArrayBuffer.isView(description)
    ? new Uint8Array(description.buffer, description.byteOffset, description.byteLength)
    : new Uint8Array(description);
  const decoder = new HevcDecoder();
  const rotation = await track.getRotation();
  const displayWidth = await track.getDisplayWidth(), displayHeight = await track.getDisplayHeight();
  const durations = new Map<number, number>();
  let lastTimestamp = -Infinity;
  const sampleFor = (frame: HdrDecodedFrame) => {
    const expectedTransfer = color.transfer === "hlg" ? 18 : 16;
    if ((frame.primaries !== 2 && frame.primaries !== 9) || (frame.transfer !== 2 && frame.transfer !== expectedTransfer)
      || (frame.matrix !== 2 && frame.matrix !== 9) || (frame.range !== 0 && frame.range !== (color.fullRange ? 2 : 1))) {
      throw new Error("HEVC frame color metadata contradicts the input HDR color space.");
    }
    if (!Number.isFinite(frame.timestamp) || frame.timestamp <= lastTimestamp) throw new Error("Invalid HDR frame presentation order.");
    lastTimestamp = frame.timestamp;
    const key = Math.round(frame.timestamp * 1e6);
    const duration = durations.get(key) ?? frame.duration;
    durations.delete(key);
    if (!(duration > 0)) throw new Error("Missing HDR frame duration.");
    return new VideoSample(frame.pixels, {
      format: "I420P10", codedWidth: frame.width, codedHeight: frame.height,
      timestamp: frame.timestamp, duration, colorSpace: webColor(color), rotation, displayWidth, displayHeight,
    });
  };
  try {
    await decoder.request({ kind: "open", url: runtimeAssetUrl("hdr-hevc.mjs"), extra });
    const sink = new EncodedPacketSink(track);
    const first = await sink.getKeyPacket(start, { verifyKeyPackets: true }) ?? await sink.getFirstKeyPacket({ verifyKeyPackets: true });
    if (!first) throw new Error("No decodable HDR keyframe was found.");
    for await (const packet of sink.packets(first)) {
      durations.set(Math.round(packet.timestamp * 1e6), packet.duration);
      await decoder.request({ kind: "send", data: packet.data, timestamp: packet.timestamp, duration: packet.duration });
      for (;;) {
        const { frame } = await decoder.request({ kind: "receive" });
        if (!frame) break;
        if (frame.timestamp >= end) return;
        const sample = sampleFor(frame);
        if (sample.timestamp + sample.duration > start) yield sample;
        else sample.close();
      }
    }
    await decoder.request({ kind: "send", data: null, timestamp: 0, duration: 0 });
    for (;;) {
      const { frame } = await decoder.request({ kind: "receive" });
      if (!frame || frame.timestamp >= end) break;
      const sample = sampleFor(frame);
      if (sample.timestamp + sample.duration > start) yield sample;
      else sample.close();
    }
  } finally { decoder.close(); }
}

export async function* hdrSamples(track: InputVideoTrack, color: HdrColor, start: number, end: number): AsyncGenerator<VideoSample> {
  // Use native decoding when its 10-bit planes are accessible. Some Windows
  // hardware decoders instead return opaque surfaces with no WebCodecs readback.
  const isHevc = await track.getCodec() === "hevc";
  if (isHevc) {
    let readable = false;
    try {
      const probe = await new VideoSampleSink(track, { hardwareAcceleration: "prefer-hardware" }).getSample(start);
      if (probe) {
        try {
          readable = probe.format === "I420P10";
          if (readable) {
            assertColor(probe.colorSpace, color);
            await probe.copyTo(new Uint8Array(probe.allocationSize()));
          }
        } finally { probe.close(); }
      }
    } catch { readable = false; }
    if (!readable) { yield* softwareSamples(track, color, start, end); return; }
  }
  const sink = new VideoSampleSink(track, { hardwareAcceleration: isHevc ? "prefer-hardware" : "prefer-software" });
  for await (const sample of sink.samples(start, end)) {
    try {
      if (sample.format !== "I420P10") throw new Error("This browser cannot expose the source's 10-bit HDR pixels.");
      assertColor(sample.colorSpace, color);
    } catch (error) { sample.close(); throw error; }
    yield sample;
  }
}

export async function hdrOutputSample(sample: VideoSample, color: HdrColor, timestamp: number, duration: number, mask?: HdrMask): Promise<VideoSample> {
  assertColor(sample.colorSpace, color);
  if (sample.format !== "I420P10") throw new Error("HDR export requires 10-bit pixel data.");
  const width = sample.codedWidth, height = sample.codedHeight;
  const cw = Math.ceil(width / 2), ch = Math.ceil(height / 2);
  const layout = [
    { offset: 0, stride: width * 2 },
    { offset: width * height * 2, stride: cw * 2 },
    { offset: (width * height + cw * ch) * 2, stride: cw * 2 },
  ];
  const pixels = new Uint16Array(width * height + 2 * cw * ch);
  await sample.copyTo(pixels, { layout });
  if (mask) compositeHdrScore(pixels, width, height, mask, sample.rotation, sample.displayWidth, sample.displayHeight, color.transfer, color.fullRange);
  return new VideoSample(pixels, {
    format: "I420P10", codedWidth: width, codedHeight: height, timestamp, duration, colorSpace: webColor(color),
    rotation: sample.rotation, displayWidth: sample.displayWidth, displayHeight: sample.displayHeight,
  });
}
