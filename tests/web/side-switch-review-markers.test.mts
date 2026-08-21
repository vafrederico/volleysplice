import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { POST } from "../../app/api/side-switch-review/markers/route.ts";
import {
  getFullVideoSideSwitchMarkerPath,
  loadFullVideoSideSwitchMarkerState,
  SideSwitchReviewValidationError,
  saveFullVideoSideSwitchMarkers,
} from "../../lib/server/side-switch-review.ts";

const MARKER_KIND = "volleycut-full-video-side-switch-markers-v1";

function json(value: unknown): string {
  return `${JSON.stringify(value, null, 2)}\n`;
}

test("full-video markers persist separately and validate report-bound coverage", async (context) => {
  const directory = await mkdtemp(
    path.join(os.tmpdir(), "volleycut-switch-markers-"),
  );
  const reportPath = path.join(directory, "report.json");
  const markerPath = path.join(directory, "markers.json");
  const decisionPath = path.join(directory, "decisions.json");
  const previousReport = process.env.VOLLEYCUT_SIDE_SWITCH_REPORT;
  const previousMarkers = process.env.VOLLEYCUT_SIDE_SWITCH_MARKERS;
  const previousDecisions = process.env.VOLLEYCUT_SIDE_SWITCH_DECISIONS;
  process.env.VOLLEYCUT_SIDE_SWITCH_REPORT = reportPath;
  process.env.VOLLEYCUT_SIDE_SWITCH_MARKERS = markerPath;
  process.env.VOLLEYCUT_SIDE_SWITCH_DECISIONS = decisionPath;
  context.after(async () => {
    if (previousReport === undefined)
      delete process.env.VOLLEYCUT_SIDE_SWITCH_REPORT;
    else process.env.VOLLEYCUT_SIDE_SWITCH_REPORT = previousReport;
    if (previousMarkers === undefined)
      delete process.env.VOLLEYCUT_SIDE_SWITCH_MARKERS;
    else process.env.VOLLEYCUT_SIDE_SWITCH_MARKERS = previousMarkers;
    if (previousDecisions === undefined)
      delete process.env.VOLLEYCUT_SIDE_SWITCH_DECISIONS;
    else process.env.VOLLEYCUT_SIDE_SWITCH_DECISIONS = previousDecisions;
    await rm(directory, { recursive: true, force: true });
  });

  await Promise.all([
    writeFile(
      reportPath,
      json({
        kind: "appearance-fixture",
        createdAt: "2026-08-21T00:00:00.000Z",
        summary: {},
        labels: {
          files: [
            { recordingId: "game-a", durationSeconds: 120 },
            { recordingId: "game-b", durationSeconds: 90 },
          ],
        },
        events: [
          {
            eventId: "game-a:gap:1",
            recordingId: "game-a",
            gapEnd: 30,
          },
          {
            eventId: "game-b:gap:1",
            recordingId: "game-b",
            gapEnd: 40,
          },
        ],
      }),
    ),
    writeFile(decisionPath, "candidate decisions stay immutable\n"),
  ]);

  assert.equal(getFullVideoSideSwitchMarkerPath(), markerPath);
  assert.deepEqual((await loadFullVideoSideSwitchMarkerState()).markers, []);

  const requestBody = {
    schemaVersion: 1,
    kind: MARKER_KIND,
    reportKind: "appearance-fixture",
    reportCreatedAt: "2026-08-21T00:00:00.000Z",
    markers: [
      {
        id: "manual:game-b:second",
        recordingId: "game-b",
        time: 42.1236,
        createdAt: "2026-08-21T01:00:00.000Z",
      },
      {
        id: "manual:game-a:first",
        recordingId: "game-a",
        time: 21.25,
        createdAt: "2026-08-21T01:01:00.000Z",
      },
    ],
    reviewedRecordingIds: ["game-b", "game-a"],
  };
  const saved = await saveFullVideoSideSwitchMarkers(requestBody);
  assert.deepEqual(
    saved.markers.map((marker) => [marker.recordingId, marker.time]),
    [
      ["game-a", 21.25],
      ["game-b", 42.124],
    ],
  );
  assert.deepEqual(saved.reviewedRecordingIds, ["game-a", "game-b"]);
  assert.equal(
    await readFile(decisionPath, "utf8"),
    "candidate decisions stay immutable\n",
  );
  assert.deepEqual(
    (await loadFullVideoSideSwitchMarkerState()).reviewedRecordingIds,
    ["game-a", "game-b"],
  );

  await assert.rejects(
    saveFullVideoSideSwitchMarkers({
      ...requestBody,
      markers: [
        {
          ...requestBody.markers[0],
          recordingId: "unknown-game",
        },
      ],
    }),
    SideSwitchReviewValidationError,
  );
  await assert.rejects(
    saveFullVideoSideSwitchMarkers({
      ...requestBody,
      markers: [{ ...requestBody.markers[0], time: 90.001 }],
    }),
    SideSwitchReviewValidationError,
  );

  const response = await POST(
    new Request("http://review.test/api/side-switch-review/markers", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: "http://review.test",
      },
      body: JSON.stringify({
        ...requestBody,
        markers: requestBody.markers.slice(0, 1),
        reviewedRecordingIds: ["game-b"],
      }),
    }),
  );
  assert.equal(response.status, 200);
  assert.equal(response.headers.get("cache-control"), "private, no-store");
  assert.deepEqual(
    ((await response.json()) as { reviewedRecordingIds: string[] })
      .reviewedRecordingIds,
    ["game-b"],
  );

  const rejectedOrigin = await POST(
    new Request("http://review.test/api/side-switch-review/markers", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: "http://attacker.test",
      },
      body: JSON.stringify(requestBody),
    }),
  );
  assert.equal(rejectedOrigin.status, 403);
});
