import assert from "node:assert/strict";
import test from "node:test";

import { parseLabelDocument } from "../../lib/annotations.ts";

function fixture() {
  return {
    schemaVersion: 1,
    kind: "volleycut-rally-labels",
    createdAt: "2026-08-12T00:00:00Z",
    recording: {
      id: "fixture",
      video: "fixture.mp4",
      videoFilename: "fixture.mp4",
      contentSha256: "0".repeat(64),
      durationSeconds: 60,
      sourceGroup: "fixture-source",
      split: "train",
      environment: "grass",
      game: { playersPerTeam: 2, targetPoints: 21, format: "2v2" },
      capture: {},
      roi: null,
    },
    annotationPolicy: {
      id: "serve-contact-to-dead-ball-v1",
      rallyStart: "serve-ball contact",
      rallyEnd: "first instant live play has ended",
      intervalConvention: "half-open",
    },
    annotation: {
      status: "in-progress",
      annotator: "reviewer",
      continuousVideoReviewed: false,
      reviewedAt: null,
      notes: "",
    },
    rallies: [{ start: 10, end: 20, tags: [] as string[] }],
    ignoredIntervals: [] as Array<Record<string, unknown>>,
    hardNegatives: [] as Array<Record<string, unknown>>,
  };
}

test("legacy label documents remain valid without optional experiment fields", () => {
  const parsed = parseLabelDocument(fixture());
  assert.deepEqual(parsed.serveMarkers, []);
  assert.deepEqual(parsed.sideSwitches, []);
  assert.equal(parsed.recording.courtGeometry, undefined);
  assert.equal(parsed.rallies[0].terminalCue, undefined);
});

test("ignored spans can overlay rally and hard-negative labels", () => {
  const value = fixture();
  value.ignoredIntervals = [{ start: 15, end: 35, reason: "ambiguous" }];
  value.hardNegatives = [
    { start: 30, end: 40, category: "foreground-crossing" },
  ];
  const parsed = parseLabelDocument(value);
  assert.deepEqual(parsed.ignoredIntervals, value.ignoredIntervals);
});

test("parser accepts editable model serve and side-switch markers", () => {
  const value = fixture();
  Object.assign(value, {
    serveMarkers: [
      {
        time: 10,
        side: "near",
        origin: "model",
        modelSide: "near",
        modelConfidence: 0.91,
        modelId: "serving-side-fixture",
        rallyId: "R001",
      },
      { time: 30, side: "review", origin: "manual" },
    ],
    sideSwitches: [
      {
        time: 25,
        origin: "model",
        modelConfidence: 0.82,
        modelId: "side-switch-fixture",
        modelEventId: "switch:1",
      },
    ],
  });

  const parsed = parseLabelDocument(value);
  assert.equal(parsed.serveMarkers[0].modelSide, "near");
  assert.equal(parsed.serveMarkers[1].side, "review");
  assert.equal(parsed.sideSwitches[0].modelEventId, "switch:1");
});

test("parser rejects invalid serve marker labels and confidence", () => {
  const badSide = fixture();
  Object.assign(badSide, { serveMarkers: [{ time: 10, side: "left" }] });
  assert.throws(() => parseLabelDocument(badSide), /serveMarkers\[0\]/);

  const badConfidence = fixture();
  Object.assign(badConfidence, {
    serveMarkers: [{ time: 10, side: "near", modelConfidence: 2 }],
  });
  assert.throws(() => parseLabelDocument(badConfidence), /serveMarkers\[0\]/);
});

test("parser accepts normalized court anchors and bounded transition cues", () => {
  const value = fixture();
  Object.assign(value.recording, {
    courtGeometry: {
      corners: {
        nearLeft: { x: 0.1, y: 0.9 },
        nearRight: { x: 0.9, y: 0.9 },
        farLeft: { x: 0.35, y: 0.2 },
        farRight: { x: 0.65, y: 0.2 },
      },
      netAnchors: { left: { x: 0.25, y: 0.55 }, right: { x: 0.75, y: 0.55 } },
      serviceZoneAnchors: { near: { x: 0.5, y: 0.95 }, far: { x: 0.5, y: 0.12 } },
    },
  });
  Object.assign(value.rallies[0], {
    receiverReactionTime: 10.4,
    collectiveStandDownTime: 20.2,
    terminalCue: "no-recovery",
    endObservability: "partially-observable",
    startConfidence: 0.9,
    endConfidence: 0.7,
    verifiedImmediateResult: true,
    playerTracklets: [
      {
        trackId: "P1",
        window: "serve",
        team: "team-a",
        courtSide: "near",
        observations: [
          { time: 9.8, footpoint: { x: 0.4, y: 0.8 }, state: "ready" },
          {
            time: 10.2,
            box: { x: 0.35, y: 0.4, width: 0.1, height: 0.4 },
            state: "playing",
          },
        ],
      },
    ],
  });
  value.hardNegatives = [
    { start: 30, end: 32, category: "walking-ball-retrieval" },
    { start: 34, end: 36, category: "celebration-huddle" },
    { start: 40, end: 42, category: "random-dead-control" },
  ];

  const parsed = parseLabelDocument(value);
  assert.equal(parsed.recording.courtGeometry?.corners.farRight?.x, 0.65);
  assert.equal(parsed.rallies[0].terminalCue, "no-recovery");
  assert.equal(parsed.rallies[0].playerTracklets?.[0].observations.length, 2);
  assert.equal(parsed.hardNegatives[2].category, "random-dead-control");
});

test("parser rejects invalid normalized points, categories, and transition timing", () => {
  const badPoint = fixture();
  Object.assign(badPoint.recording, {
    courtGeometry: { corners: { nearLeft: { x: 1.1, y: 0.9 } } },
  });
  assert.throws(() => parseLabelDocument(badPoint), /normalized finite x and y/);

  const badCategory = fixture();
  badCategory.hardNegatives = [{ start: 30, end: 32, category: "misc" }];
  assert.throws(() => parseLabelDocument(badCategory), /category is invalid/);

  const badTransition = fixture();
  Object.assign(badTransition.rallies[0], { receiverReactionTime: 16 });
  assert.throws(() => parseLabelDocument(badTransition), /within five seconds after rally start/);

  const badConfidence = fixture();
  Object.assign(badConfidence.rallies[0], { endConfidence: -0.1 });
  assert.throws(() => parseLabelDocument(badConfidence), /endConfidence must be between 0 and 1/);

  const partialCompleteGeometry = fixture();
  partialCompleteGeometry.annotation.status = "complete";
  Object.assign(partialCompleteGeometry.recording, {
    courtGeometry: { corners: { nearLeft: { x: 0.1, y: 0.9 } } },
  });
  assert.throws(
    () => parseLabelDocument(partialCompleteGeometry),
    /all four named corners/,
  );
});

test("parser rejects identified, out-of-window, and malformed player tracklets", () => {
  const identified = fixture();
  Object.assign(identified.rallies[0], {
    playerTracklets: [
      {
        trackId: "P1",
        window: "serve",
        team: "team-a",
        courtSide: "near",
        playerName: "must-not-be-stored",
        observations: [{ time: 10.1, footpoint: { x: 0.4, y: 0.8 } }],
      },
    ],
  });
  assert.throws(() => parseLabelDocument(identified), /playerName is not recognized/);

  const namedToken = fixture();
  Object.assign(namedToken.rallies[0], {
    playerTracklets: [
      {
        trackId: "ALICE",
        window: "serve",
        team: "team-a",
        courtSide: "near",
        observations: [{ time: 10.1, footpoint: { x: 0.4, y: 0.8 } }],
      },
    ],
  });
  assert.throws(() => parseLabelDocument(namedToken), /anonymous token/);

  const outOfWindow = fixture();
  Object.assign(outOfWindow.rallies[0], {
    playerTracklets: [
      {
        trackId: "P1",
        window: "serve",
        team: "team-a",
        courtSide: "near",
        observations: [{ time: 15, footpoint: { x: 0.4, y: 0.8 } }],
      },
    ],
  });
  assert.throws(() => parseLabelDocument(outOfWindow), /inside its boundary window/);

  const badBox = fixture();
  Object.assign(badBox.rallies[0], {
    playerTracklets: [
      {
        trackId: "P1",
        window: "rally-end",
        team: "team-b",
        courtSide: "far",
        observations: [
          { time: 20.1, box: { x: 0.9, y: 0.4, width: 0.2, height: 0.4 } },
        ],
      },
    ],
  });
  assert.throws(() => parseLabelDocument(badBox), /positive normalized frame box/);
});
