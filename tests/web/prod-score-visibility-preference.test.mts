import assert from "node:assert/strict";
import test, { type TestContext } from "node:test";
import { readScoreVisibilityPreference, saveScoreVisibilityPreference } from "../../prod/src/lib/score-visibility-preference.ts";

function storage(t: TestContext, value: unknown) {
  const original = Object.getOwnPropertyDescriptor(globalThis, "localStorage");
  Object.defineProperty(globalThis, "localStorage", { configurable: true, value });
  t.after(() => {
    if (original) Object.defineProperty(globalThis, "localStorage", original);
    else Reflect.deleteProperty(globalThis, "localStorage");
  });
}

test("score visibility is reused by later exports and survives a new settings reader", (t) => {
  const data = new Map<string, string>();
  storage(t, {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => data.set(key, value),
  });
  assert.equal(readScoreVisibilityPreference(), false);
  saveScoreVisibilityPreference(true);
  assert.equal(readScoreVisibilityPreference(false), true);
  saveScoreVisibilityPreference(false);
  assert.equal(readScoreVisibilityPreference(true), false);
});

test("unavailable local storage does not prevent an export", (t) => {
  storage(t, {
    getItem() { throw new Error("Storage blocked"); },
    setItem() { throw new Error("Storage blocked"); },
  });
  assert.equal(readScoreVisibilityPreference(true), true);
  assert.doesNotThrow(() => saveScoreVisibilityPreference(true));
});
