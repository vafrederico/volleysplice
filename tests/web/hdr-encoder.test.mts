import assert from "node:assert/strict";
import { test } from "node:test";
import { chooseHdrEncoder, type HdrColor } from "../../prod/src/lib/on-device/hdr-export.ts";

const color: HdrColor = { primaries: "bt2020", transfer: "hlg", matrix: "bt2020-ncl", fullRange: false };

test("HDR encoder selection probes actual frames and preserves working native paths", async (t) => {
  const attempts: string[] = [];
  let rejectNative = false;
  let rejectHevc = false;
  let wrongColor = false;
  let closedFrames = 0;
  let closedEncoders = 0;
  class Frame {
    native = false;
    close() { closedFrames++; }
  }
  class Encoder {
    state = "configured";
    config: any;
    callbacks: any;
    constructor(callbacks: any) { this.callbacks = callbacks; }
    static async isConfigSupported() { return { supported: true }; }
    configure(config: any) { this.config = config; }
    encode(frame: Frame) {
      attempts.push(`${this.config.codec}:${frame.native ? "native" : "cpu"}`);
      if ((rejectNative && frame.native) || (rejectHevc && this.config.codec.startsWith("hvc1"))) throw new Error("Unsupported frame");
      this.callbacks.output({}, { decoderConfig: { codec: this.config.codec, colorSpace: wrongColor ? { ...color, transfer: "bt709" } : color } });
    }
    async flush() {}
    close() { closedEncoders++; this.state = "closed"; }
  }
  for (const [name, value] of [["VideoFrame", Frame], ["VideoEncoder", Encoder]] as const) {
    const original = Object.getOwnPropertyDescriptor(globalThis, name);
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
    t.after(() => {
      if (original) Object.defineProperty(globalThis, name, original);
      else Reflect.deleteProperty(globalThis, name);
    });
  }
  const source = { colorSpace: color, toVideoFrame() { const frame = new Frame(); frame.native = true; return frame; } } as any;

  assert.equal((await chooseHdrEncoder(4, 4, color, source)).direct, true);
  assert.deepEqual(attempts.splice(0), ["hvc1.2.4.L153.B0:native"]);

  rejectNative = true;
  assert.equal((await chooseHdrEncoder(4, 4, color, source)).direct, false);
  assert.deepEqual(attempts.splice(0), ["hvc1.2.4.L153.B0:native", "hvc1.2.4.L153.B0:cpu"]);

  const scored = await chooseHdrEncoder(4, 4, color);
  assert.equal(scored.direct, false);
  assert.equal(scored.encoding.codec, "hevc");
  assert.deepEqual(attempts.splice(0), ["hvc1.2.4.L153.B0:cpu"]);

  rejectHevc = true;
  assert.equal((await chooseHdrEncoder(4, 4, color)).encoding.codec, "vp9");
  assert.deepEqual(attempts.splice(0), ["hvc1.2.4.L153.B0:cpu", "vp09.02.51.10:cpu"]);

  wrongColor = true;
  await assert.rejects(chooseHdrEncoder(4, 4, color), /no working 10-bit HDR encoder/);
  assert.equal(closedFrames, 8);
  assert.equal(closedEncoders, 8);
});
