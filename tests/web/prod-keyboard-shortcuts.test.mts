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
    ["ArrowLeft", "back", "previousRally"], ["ArrowRight", "forward", "nextRally"],
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
  assert.equal(key("s", { ctrlKey: true, shiftKey: true }), "setStart");
  assert.equal(key("S", { ctrlKey: true, shiftKey: true }), "setStart");
  assert.equal(key("e", { ctrlKey: true, shiftKey: true }), "setEnd");
  assert.equal(key("E", { ctrlKey: true, shiftKey: true }), "setEnd");
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

test("R interleaves clips, serves, and cleanup by time before wrapping, without modifying the queue", () => {
  let cursor: ReviewItem | null = null;
  const visited = [];
  for (let i = 0; i < 6; i++) {
    cursor = nextReviewItem(items, cursor, cursor?.time ?? 0);
    visited.push(cursor?.id);
  }
  assert.deepEqual(visited, ["c1", "s1", "x1", "s2", "x2", "c1"]);
  assert.equal(items[0].id, "s2");
});

test("review progression survives resolving the current item and skips empty queues", () => {
  const previous = items.find((item) => item.id === "x1")!;
  assert.equal(nextReviewItem(items.filter((item) => item.id !== previous.id), previous, previous.time)?.id, "s2");
  assert.equal(nextReviewItem(items.filter((item) => item.kind === "serve"), null, 0)?.id, "s1");
  assert.equal(nextReviewItem([], previous, previous.time), null);
  assert.equal(nextReviewItem([], null, 0), null);
  const withoutServes = items.filter((item) => item.kind !== "serve");
  assert.equal(nextReviewItem(withoutServes, items[0], 50)?.id, "x2");
});

test("R uses the new playhead after seeking backward or forward", () => {
  assert.equal(nextReviewItem(items, null, 0)?.id, "c1");
  assert.equal(nextReviewItem(items, null, 1)?.id, "s1");
  assert.equal(nextReviewItem(items, null, 5)?.id, "x1");
  assert.equal(nextReviewItem(items, null, 20)?.id, "s2");
  assert.equal(nextReviewItem(items, null, 95)?.id, "c1");
  assert.equal(nextReviewItem(items, items[1], 20)?.id, "s2", "playback can advance beyond the opened item");
});

test("R skips the opened rally and earlier items in its padding, even after resolving it", () => {
  const current = items.find((item) => item.id === "x1")!;
  const paddedItems: ReviewItem[] = [...items, { kind: "serve", id: "in-padding", time: 9 }];
  assert.equal(nextReviewItem(paddedItems, current, 8)?.id, "s2");
  assert.equal(nextReviewItem(paddedItems.filter(item => item.id !== current.id), current, 8)?.id, "s2");
});

test("R visits each item at a shared timestamp, including clip and serve IDs that collide", () => {
  const tied: ReviewItem[] = [
    { kind: "serve", id: "b", time: 10 },
    { kind: "clip", id: "same", time: 10 },
    { kind: "cleanup", id: "cleanup", time: 10 },
    { kind: "serve", id: "same", time: 10 },
    { kind: "clip", id: "later", time: 20 },
  ];
  let cursor: ReviewItem | null = null;
  const visited = [];
  for (let i = 0; i < 6; i++) {
    cursor = nextReviewItem(tied, cursor, cursor?.time ?? 0);
    visited.push([cursor?.kind, cursor?.id]);
  }
  assert.deepEqual(visited, [
    ["cleanup", "cleanup"], ["clip", "same"], ["serve", "b"],
    ["serve", "same"], ["clip", "later"], ["cleanup", "cleanup"],
  ]);
  const clip = tied[1];
  assert.equal(nextReviewItem(tied.filter(item => item !== clip), clip, 8)?.id, "b");
});
