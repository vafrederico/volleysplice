import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const styles = await readFile(
  new URL("../../prod/src/designs/taste/taste-designs.css", import.meta.url),
  "utf8",
);

test("mobile Rally Desk reserves a scrollable viewport for the serve list", () => {
  assert.match(
    styles,
    /\.rd-mobile-events > \.rd-events > \.rd-event-list \{\s*height: clamp\(112px, 18dvh, 160px\);\s*min-height: 112px;\s*flex: none;\s*\}/,
  );
  assert.match(
    styles,
    /\.rd-mobile-events > \.rd-events \{ height: auto;[^}]+\}/,
  );
});
