import assert from "node:assert/strict";
import test from "node:test";
import { nextReviewItem, shortcutAction, type ReviewItem } from "../../prod/src/designs/taste/keyboard-shortcuts.ts";

const key = (value: string, overrides = {}) => shortcutAction({
  key: value, shiftKey: false, ctrlKey: false, metaKey: false,
  altKey: false, repeat: false, isComposing: false, ...overrides,
});

test("review shortcuts distinguish shifted actions including serve and split", () => {
  for (const [value, plain, shifted] of [
    ["n", "near", "new"], ["r", "review", "missed"],
    ["e", "exclude", "export"], ["s", "serve", "split"],
    ["Backspace", "remove", "removeEvent"],
  ]) {
    assert.equal(key(value), plain);
    assert.equal(key(value.toUpperCase(), { shiftKey: true }), shifted);
  }
  for (const [value, action] of [
    ["f", "far"], ["k", "keep"], ["t", "switch"], ["p", "projects"],
    ["ArrowLeft", "back"], ["ArrowRight", "forward"], ["-", "slower"],
    ["=", "faster"], ["Escape", "cancelRange"], [" ", "playPause"],
  ]) assert.equal(key(value), action);
  assert.equal(key("+", { shiftKey: true }), "faster");
  assert.equal(key("+"), "faster");
  assert.equal(key("F"), "far");
  assert.equal(key("f", { shiftKey: true }), null);
  assert.equal(key("z", { ctrlKey: true }), "undo");
  assert.equal(key("y", { ctrlKey: true }), "redo");
  assert.equal(key("z"), null);
  assert.equal(key("y"), null);
  assert.equal(key("z", { ctrlKey: true, shiftKey: true }), null);
  assert.equal(key("z", { ctrlKey: true, repeat: true }), null);
});

test("browser shortcuts, composition, and held edit keys do not cause edits", () => {
  for (const flag of ["ctrlKey", "metaKey", "altKey", "isComposing", "repeat"]) {
    for (const value of ["Backspace", "n", "s", "r", "e", "t", "k", "+", "Escape", " "])
      assert.equal(key(value, { [flag]: true }), null);
  }
  assert.equal(key("ArrowLeft", { repeat: true }), "back");
  assert.equal(key("ArrowRight", { repeat: true }), "forward");
  assert.equal(key("Escape", { shiftKey: true }), null);
});

const items: ReviewItem[] = [
  { kind: "serve", id: "s2", time: 50 },
  { kind: "clip", id: "c1", time: 1 },
  { kind: "cleanup", id: "x2", time: 90 },
  { kind: "serve", id: "s1", time: 2 },
  { kind: "cleanup", id: "x1", time: 10 },
];

test("R starts with cleanup and visits every item before wrapping, without modifying the queue", () => {
  let cursor: ReviewItem | null = null;
  const visited = [];
  for (let i = 0; i < 6; i++) {
    cursor = nextReviewItem(items, cursor);
    visited.push(cursor?.id);
  }
  assert.deepEqual(visited, ["x1", "x2", "c1", "s1", "s2", "x1"]);
  assert.equal(items[0].id, "s2");
});

test("review progression survives resolving the current item and skips empty queues", () => {
  const previous = items.find((item) => item.id === "x2")!;
  assert.equal(nextReviewItem(items.filter((item) => item.id !== "x2"), previous)?.id, "c1");
  assert.equal(nextReviewItem(items.filter((item) => item.kind === "serve"), null)?.id, "s1");
  assert.equal(nextReviewItem([], previous), null);
  assert.equal(nextReviewItem([], null), null);
});
