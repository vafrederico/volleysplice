import assert from "node:assert/strict";
import test from "node:test";

import { requestPlayingSeek as requestLocalPlayingSeek } from "../../lib/on-device/player.ts";
import { requestPlayingSeek as requestProdPlayingSeek } from "../../prod/src/lib/on-device/player.ts";

const implementations = [
  ["local", requestLocalPlayingSeek],
  ["prod", requestProdPlayingSeek],
] as const;

function fakeVideo(options?: { currentTime?: number; duration?: number; seeking?: boolean }) {
  let playCalls = 0;
  return {
    video: {
      currentTime: options?.currentTime ?? 4,
      duration: options?.duration ?? 20,
      seeking: options?.seeking ?? false,
      play: async () => {
        playCalls += 1;
      },
    },
    playCalls: () => playCalls,
  };
}

for (const [app, requestPlayingSeek] of implementations) {
  test(`${app}: a real seek waits for seeked before requesting playback`, () => {
    const { video, playCalls } = fakeVideo();

    assert.equal(requestPlayingSeek(video, 12), true);
    assert.equal(video.currentTime, 12);
    assert.equal(playCalls(), 0);
  });

  test(`${app}: seeks are clamped to the playable timeline`, () => {
    const { video } = fakeVideo();

    requestPlayingSeek(video, 30);
    assert.equal(video.currentTime, 20);

    requestPlayingSeek(video, -3);
    assert.equal(video.currentTime, 0);
  });

  test(`${app}: selecting the current frame plays immediately because no seeked event is expected`, async () => {
    const { video, playCalls } = fakeVideo({ currentTime: 4.005 });

    assert.equal(requestPlayingSeek(video, 4), false);
    await Promise.resolve();
    assert.equal(playCalls(), 1);
  });
}
