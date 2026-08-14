type SeekableVideo = Pick<
  HTMLVideoElement,
  "currentTime" | "duration" | "seeking" | "play"
>;

/**
 * Moves a playing preview without asking the decoder to play until its seek has completed.
 * Returns true when callers should wait for the media element's `seeked` event to resume.
 */
export function requestPlayingSeek(video: SeekableVideo, seconds: number): boolean {
  const target = Number.isFinite(video.duration)
    ? Math.min(Math.max(0, seconds), video.duration)
    : Math.max(0, seconds);

  if (Math.abs(video.currentTime - target) < 0.01 && !video.seeking) {
    void video.play().catch(() => undefined);
    return false;
  }

  video.currentTime = target;
  return true;
}
