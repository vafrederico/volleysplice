import assert from "node:assert/strict";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { createServer } from "vite";

// Load the real component without listening on a port or serving the app.
const server = await createServer({
  root: fileURLToPath(new URL("..", import.meta.url)),
  server: { middlewareMode: true, hmr: false },
  appType: "custom",
});
const dom = new JSDOM('<div id="root"></div>', { url: "https://example.test" });
Object.assign(globalThis, {
  window: dom.window,
  document: dom.window.document,
  IS_REACT_ACT_ENVIRONMENT: true,
});
Object.defineProperty(globalThis, "navigator", { value: dom.window.navigator, configurable: true });
dom.window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
let copied = "";
let download;
let downloadedBlob;
Object.defineProperty(navigator, "clipboard", { value: { writeText: async (text) => { copied = text; } } });
const originalCreateURL = URL.createObjectURL;
const originalRevokeURL = URL.revokeObjectURL;
URL.createObjectURL = (blob) => { downloadedBlob = blob; return "blob:chapters"; };
URL.revokeObjectURL = () => {};
dom.window.HTMLAnchorElement.prototype.click = function () { download = this.download; };
const root = createRoot(document.getElementById("root"));

try {
  const { YouTubeChaptersModal } = await server.ssrLoadModule("/src/components/YouTubeChaptersModal.tsx");
  const { createScoreTracking, addServeMarker } = await server.ssrLoadModule("/src/lib/score-tracking.ts");
  const cuts = [10, 40].map((start, index) => ({
    id: `R${index + 1}`, coreStart: start, coreEnd: start + 10,
    keepStart: start, keepEnd: start + 10, confidence: 1, included: true, origin: "manual",
  }));
  const props = {
    sourceFilename: "Match.mp4", cuts,
    intervals: cuts.map((cut) => ({ start: cut.keepStart, end: cut.keepEnd, cutIds: [cut.id] })),
    scoreTracking: addServeMarker(createScoreTracking(), 42, "near", { id: "S2", rallyId: "R2" }),
    hasSideSwitches: false, onClose: () => {},
  };
  const click = async (element) => { assert.ok(element); await act(async () => element.click()); };
  const button = (text) => [...document.querySelectorAll("button")].find((node) => node.textContent === text);
  const credit = () => [...document.querySelectorAll("label")].find((node) => node.textContent.includes("Include VolleySplice credit")).querySelector("input");
  const plain = "0:10 0–0 - Team 1 serving";
  const credited = `Edited with https://volleysplice.com\n\n${plain}`;
  await act(async () => root.render(React.createElement(YouTubeChaptersModal, props)));
  assert.equal(credit().checked, true);
  assert.equal(document.querySelector("pre").textContent, credited);
  await click(button("Copy chapters"));
  assert.equal(copied, credited);
  await click(document.querySelector('input[value="text-file"]'));
  await click(button("Download text file"));
  assert.equal(download, "Match-youtube-chapters.txt");
  assert.equal(await downloadedBlob.text(), `${credited}\n`);
  await click(credit());
  assert.equal(credit().checked, false);
  assert.equal(document.querySelector("pre").textContent, plain);
  await click(button("Download text file"));
  assert.equal(await downloadedBlob.text(), `${plain}\n`);
  await click(document.querySelector('input[value="clipboard"]'));
  await click(button("Copy chapters"));
  assert.equal(copied, plain);
  await act(async () => root.render(React.createElement(YouTubeChaptersModal, { ...props, scoreTracking: createScoreTracking() })));
  assert.equal(button("Copy chapters").disabled, true);
  assert.equal(document.querySelector("pre").textContent, "No visible rallies are available for chapter export.");
  console.log("YouTube dialog: default credit, toggle, preview, clipboard, download, serve filtering, and empty export passed.");
} finally {
  await act(async () => root.unmount());
  await server.close();
  URL.createObjectURL = originalCreateURL;
  URL.revokeObjectURL = originalRevokeURL;
  dom.window.close();
}
