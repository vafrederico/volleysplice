import assert from "node:assert/strict";
import test from "node:test";
import {
  clampLayout,
  readLayout,
} from "../../prod/src/designs/taste/workspace-layout.ts";

test("stored layout tolerates malformed and invalid preferences", () => {
  for (const input of [null, "broken", "null", "[]", "42"])
    assert.deepEqual(readLayout(input), {});
  assert.deepEqual(
    readLayout('{"left":360,"right":-1,"video":"400","extra":123}'),
    { left: 360 },
  );
  assert.deepEqual(readLayout('{"left":1e999,"video":420}'), { video: 420 });
});

test("scaled desktop viewports retain usable center space without changing saved sizes", () => {
  const saved = { left: 480, right: 480, video: 800 };
  for (const width of [1321, 1366, 1536, 1920]) {
    const layout = clampLayout(saved, width, 620);
    assert.ok(width - layout.left - layout.right >= 480);
    assert.ok(layout.left >= 240 && layout.left <= layout.leftMax);
    assert.ok(layout.right >= 260 && layout.right <= layout.rightMax);
    assert.ok(layout.video <= 434);
  }
  assert.deepEqual(saved, { left: 480, right: 480, video: 800 });
  assert.equal(clampLayout(saved, 1920, 1200).video, 800);
});

test("two-column and mobile layouts ignore hidden sidebar space", () => {
  assert.equal(clampLayout({}, 1000, 900).video, ((1000 - 275 - 36) * 9) / 16);
  assert.equal(clampLayout({}, 800, 900).video, ((800 - 36) * 9) / 16);
  assert.equal(clampLayout({}, 390, 844).video, ((390 - 28) * 9) / 16);
});
