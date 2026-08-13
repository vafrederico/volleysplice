import {
  ALL_FORMATS,
  BlobSource,
  Input,
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

export async function openLocalMedia(file: File): Promise<OpenedMedia> {
  const input = new Input({
    source: new BlobSource(file, { maxCacheSize: 8 * 1024 * 1024 }),
    formats: ALL_FORMATS,
  });
  try {
    if (!(await input.canRead())) {
      throw new Error("This browser cannot read the selected media container.");
    }

    const videoTrack = await input.getPrimaryVideoTrack();
    if (!videoTrack) {
      throw new Error("The selected file has no video track.");
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
    if (!info.canDecodeVideo) {
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

export function* analysisTimestamps(duration: number, fps: number) {
  const count = Math.max(1, Math.ceil(duration * fps));
  for (let index = 0; index < count; index += 1) {
    const timestamp = index / fps;
    if (timestamp >= duration) break;
    yield timestamp;
  }
}
