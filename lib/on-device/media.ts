import {
  ALL_FORMATS,
  BlobSource,
  Input,
  UrlSource,
  type InputAudioTrack,
  type InputVideoTrack,
} from "mediabunny";

import type { OnDeviceMediaInfo } from "./types";

export type OpenedMedia = {
  input: Input;
  videoTrack: InputVideoTrack;
  audioTrack: InputAudioTrack | null;
  info: OnDeviceMediaInfo;
};

const MAX_SOURCE_CACHE_SIZE = 8 * 1024 * 1024;

type ProbeMessages = { unreadable: string; noVideo: string };

async function probeMedia(
  input: Input,
  messages: ProbeMessages,
  requireDecodableVideo = true,
): Promise<OpenedMedia> {
  try {
    if (!(await input.canRead())) {
      throw new Error(messages.unreadable);
    }

    const videoTrack = await input.getPrimaryVideoTrack();
    if (!videoTrack) {
      throw new Error(messages.noVideo);
    }
    const audioTrack = await input.getPrimaryAudioTrack();
    const [duration, canDecodeVideo, canDecodeAudio] = await Promise.all([
      input.computeDuration(),
      videoTrack.canDecode(),
      audioTrack ? audioTrack.canDecode() : Promise.resolve(true),
    ]);
    const [width, height, rotation, videoCodec, videoCodecString] = await Promise.all([
      videoTrack.getDisplayWidth(),
      videoTrack.getDisplayHeight(),
      videoTrack.getRotation(),
      videoTrack.getCodec(),
      videoTrack.getCodecParameterString(),
    ]);
    const [audioCodec, sampleRate, channels] = audioTrack
      ? await Promise.all([
          audioTrack.getCodec(),
          audioTrack.getSampleRate(),
          audioTrack.getNumberOfChannels(),
        ])
      : [null, null, null];

    const info: OnDeviceMediaInfo = {
      duration,
      mimeType: await input.getMimeType(),
      width,
      height,
      rotation,
      videoCodec: videoCodec ?? "unknown",
      videoCodecString,
      canDecodeVideo,
      hasAudio: audioTrack !== null,
      audioCodec: audioCodec ?? null,
      sampleRate,
      channels,
      canDecodeAudio,
    };
    if (requireDecodableVideo && !info.canDecodeVideo) {
      throw new Error(
        `The container is readable, but this browser cannot decode ${info.videoCodec}.`,
      );
    }
    return { input, videoTrack, audioTrack, info };
  } catch (error) {
    input.dispose();
    throw error;
  }
}

export function openLocalMedia(file: File): Promise<OpenedMedia> {
  return probeMedia(new Input({
    source: new BlobSource(file, { maxCacheSize: MAX_SOURCE_CACHE_SIZE }),
    formats: ALL_FORMATS,
  }), {
    unreadable: "This browser cannot read the selected media container.",
    noVideo: "The selected file has no video track.",
  });
}

export function openLocalMediaForAudio(file: File): Promise<OpenedMedia> {
  return probeMedia(new Input({
    source: new BlobSource(file, { maxCacheSize: MAX_SOURCE_CACHE_SIZE }),
    formats: ALL_FORMATS,
  }), {
    unreadable: "This browser cannot read the selected media container.",
    noVideo: "The selected file has no video track.",
  }, false);
}

export function openUrlMedia(
  url: string | URL | Request,
  requestInit?: Omit<RequestInit, "signal">,
): Promise<OpenedMedia> {
  return probeMedia(new Input({
    source: new UrlSource(url, {
      maxCacheSize: MAX_SOURCE_CACHE_SIZE,
      requestInit,
    }),
    formats: ALL_FORMATS,
  }), {
    unreadable: "This browser cannot read the requested media container.",
    noVideo: "The requested media has no video track.",
  });
}

export function openUrlMediaForAudio(
  url: string | URL | Request,
  requestInit?: Omit<RequestInit, "signal">,
): Promise<OpenedMedia> {
  return probeMedia(new Input({
    source: new UrlSource(url, {
      maxCacheSize: MAX_SOURCE_CACHE_SIZE,
      requestInit,
    }),
    formats: ALL_FORMATS,
  }), {
    unreadable: "This browser cannot read the requested media container.",
    noVideo: "The requested media has no video track.",
  }, false);
}

export function* analysisTimestamps(duration: number, fps: number) {
  const count = Math.max(1, Math.ceil(duration * fps));
  for (let index = 0; index < count; index += 1) {
    const timestamp = index / fps;
    if (timestamp >= duration) break;
    yield timestamp;
  }
}
