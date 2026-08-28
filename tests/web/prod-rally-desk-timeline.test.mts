import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL("../../prod/src/designs/taste/index.tsx", import.meta.url),
  "utf8",
);
const timeline = source.slice(
  source.indexOf("type RallyDeskTimelineTarget"),
  source.indexOf("function RallyDeskSettingsModal"),
);

test("Rally Desk timeline owns pointer gestures that begin on rally and marker buttons", () => {
  assert.match(
    timeline,
    /const dragRef = useRef<RallyDeskTimelineDrag \| null>/,
  );
  assert.match(timeline, /horizontalDistance < 4/);
  assert.match(timeline, /data-timeline-cut-id=/);
  assert.match(timeline, /data-timeline-event="serve"/);
  assert.match(timeline, /onClickCapture=/);
  assert.doesNotMatch(timeline, /closest\("button"\).*return/);
  assert.equal(timeline.match(/onPointerDown=/g)?.length, 1);
});
