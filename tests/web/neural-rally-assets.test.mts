import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { verifiedAsset } from "../../prod/src/lib/on-device/rally-model.ts";

const payload = new Uint8Array([1, 2, 3, 4]);
const asset = {
  name: "encoder.onnx",
  sizeBytes: payload.length,
  sha256: createHash("sha256").update(payload).digest("hex"),
};

test("model download accepts only complete bytes matching the release manifest", async (context) => {
  context.mock.method(globalThis, "fetch", async () => new Response(payload));
  assert.deepEqual(new Uint8Array(await verifiedAsset("https://example.test/encoder.onnx", asset)), payload);
});

test("a missing selected model fails instead of returning another detector", async (context) => {
  context.mock.method(globalThis, "fetch", async () => new Response("Not found", { status: 404 }));
  await assert.rejects(verifiedAsset("https://example.test/encoder.onnx", asset), /downloaded \(404\)/);
});

test("same-length corrupted model bytes fail their release digest", async (context) => {
  context.mock.method(globalThis, "fetch", async () => new Response(new Uint8Array([1, 2, 3, 5])));
  await assert.rejects(verifiedAsset("https://example.test/encoder.onnx", asset), /integrity check/);
});

test("a truncated model fails even if the response returned success", async (context) => {
  context.mock.method(globalThis, "fetch", async () => new Response(payload.subarray(0, 3)));
  await assert.rejects(verifiedAsset("https://example.test/encoder.onnx", asset), /integrity check/);
});
