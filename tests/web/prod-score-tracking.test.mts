import assert from "node:assert/strict";
import test from "node:test";

import {
  addServeMarker,
  addSideSwitchMarker,
  createScoreTracking,
  deriveScoreAt,
  isValidScoreTracking,
  removeServeMarker,
  removeSideSwitchMarker,
  scoreBoundaryTimestamp,
  scoreTrackingOutsideExcludedRallies,
  scoreTrackingOutsideIgnoredIntervals,
  setPreviousPointIgnored,
  setServeMarkerSide,
} from "../../prod/src/lib/score-tracking.ts";

test("score tracking starts enabled with editable default team names", () => {
  const tracking = createScoreTracking();
  assert.equal(tracking.enabled, true);
  assert.equal(tracking.team1Name, "Team 1");
  assert.equal(tracking.team2Name, "Team 2");
  assert.deepEqual(deriveScoreAt(tracking), {
    team1Score: 0,
    team2Score: 0,
    servingTeamId: null,
    servingSide: null,
    points: [],
    ignoredPointCount: 0,
    reviewPointCount: 0,
  });
});

test("each serve after the first awards the preceding point to that server", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 12, "near", { id: "S002" });
  tracking = addServeMarker(tracking, 20, "far", { id: "S003" });

  assert.deepEqual(deriveScoreAt(tracking, 11), {
    team1Score: 0,
    team2Score: 0,
    servingTeamId: "team-1",
    servingSide: "near",
    points: [],
    ignoredPointCount: 0,
    reviewPointCount: 0,
  });
  assert.equal(deriveScoreAt(tracking, 12).team1Score, 1);
  assert.equal(deriveScoreAt(tracking, 19.999).team2Score, 0);
  assert.deepEqual(
    deriveScoreAt(tracking).points.map((point) => [
      point.serveMarkerId,
      point.winnerTeamId,
      point.status,
    ]),
    [
      ["S002", "team-1", "counted"],
      ["S003", "team-2", "counted"],
    ],
  );
});

test("a side switch flips physical near/far without changing team identity", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 10, "far", { id: "S002" });
  tracking = addSideSwitchMarker(tracking, 15, "X001");
  tracking = addServeMarker(tracking, 16, "far", { id: "S003" });
  tracking = addServeMarker(tracking, 20, "near", { id: "S004" });

  const atSwitch = deriveScoreAt(tracking, 15);
  assert.equal(atSwitch.team1Score, 0);
  assert.equal(atSwitch.team2Score, 1);

  const atNextServe = deriveScoreAt(tracking, 16);
  assert.equal(atNextServe.team1Score, 1);
  assert.equal(atNextServe.team2Score, 1);

  const score = deriveScoreAt(tracking);
  assert.equal(score.team1Score, 1);
  assert.equal(score.team2Score, 2);
  assert.deepEqual(
    score.points.map((point) => point.winnerTeamId),
    ["team-2", "team-1", "team-2"],
  );
});

test("replays and review verdicts stay visible but do not change the score", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 1, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 2, "far", { id: "S002" });
  tracking = setPreviousPointIgnored(tracking, "S002", true);
  tracking = addServeMarker(tracking, 3, "review", {
    id: "S003",
    origin: "model",
  });
  tracking = addServeMarker(tracking, 4, "far", { id: "S004" });

  const score = deriveScoreAt(tracking);
  assert.equal(score.team1Score, 0);
  assert.equal(score.team2Score, 1);
  assert.equal(score.ignoredPointCount, 1);
  assert.equal(score.reviewPointCount, 1);
  assert.deepEqual(
    score.points.map((point) => point.status),
    ["ignored", "review", "counted"],
  );
});

test("model verdict correction keeps its original verdict and manual markers can be removed", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 8.12345, "review", {
    id: "model-rally-1",
    origin: "model",
    rallyId: "R001",
  });
  tracking = setServeMarkerSide(tracking, "model-rally-1", "near");
  assert.equal(tracking.serveMarkers[0].timestamp, 8.123);
  assert.equal(tracking.serveMarkers[0].side, "near");
  assert.equal(tracking.serveMarkers[0].modelSide, "review");

  tracking = addServeMarker(tracking, 12, "far");
  assert.deepEqual(
    tracking.serveMarkers.map((marker) => marker.id),
    ["model-rally-1", "S001"],
  );
  tracking = removeServeMarker(tracking, "S001");
  tracking = addSideSwitchMarker(tracking, 13);
  assert.equal(tracking.sideSwitchMarkers[0].origin, "manual");
  tracking = removeSideSwitchMarker(tracking, "X001");
  assert.equal(tracking.serveMarkers.length, 1);
  assert.equal(tracking.sideSwitchMarkers.length, 0);

  tracking = removeServeMarker(tracking, "model-rally-1");
  assert.deepEqual(tracking.removedModelMarkerIds, ["model-rally-1"]);
  tracking = addServeMarker(tracking, 8.123, "near", {
    id: "model-rally-1",
    origin: "model",
  });
  assert.equal(
    tracking.serveMarkers.length,
    0,
    "cached prediction stays removed",
  );
});

test("persisted score tracking rejects invalid timestamps and duplicate marker IDs", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", { id: "same" });
  assert.equal(isValidScoreTracking(tracking, 10), true);

  const duplicate = {
    ...tracking,
    sideSwitchMarkers: [{ id: "same", timestamp: 6 }],
  };
  assert.equal(isValidScoreTracking(duplicate, 10), false);
  assert.equal(
    isValidScoreTracking(
      {
        ...tracking,
        serveMarkers: [{ ...tracking.serveMarkers[0], timestamp: 11 }],
      },
      10,
    ),
    false,
  );
});

test("ignored source sections filter points without changing stored markers", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 10, "far", { id: "S002" });
  tracking = addServeMarker(tracking, 20, "near", { id: "S003" });
  tracking = addSideSwitchMarker(tracking, 9.5, {
    id: "switch-predicted",
    origin: "model",
    modelConfidence: 0.8,
    modelEventId: "candidate-1",
    rallyIds: ["R001", "R002"],
  });

  const active = scoreTrackingOutsideIgnoredIntervals(tracking, [
    { start: 9, end: 11 },
  ]);
  assert.deepEqual(
    active.serveMarkers.map((marker) => marker.id),
    ["S001", "S003"],
  );
  assert.deepEqual(active.sideSwitchMarkers, []);
  assert.equal(deriveScoreAt(active).team1Score, 1);
  assert.equal(deriveScoreAt(active).team2Score, 0);
  assert.equal(deriveScoreAt(active).ignoredPointCount, 0);
  assert.equal(deriveScoreAt(active).points.length, 1);
  assert.equal(
    tracking.serveMarkers.length,
    3,
    "the cached/edit-layer markers stay intact",
  );

  assert.equal(
    scoreTrackingOutsideIgnoredIntervals(tracking, []).serveMarkers,
    tracking.serveMarkers,
    "removing the ignored interval restores the original marker collection",
  );
});

test("removed model side switches stay tombstoned", () => {
  let tracking = createScoreTracking();
  tracking = addSideSwitchMarker(tracking, 15, {
    id: "switch-candidate-1",
    origin: "model",
    modelConfidence: 0.75,
    modelEventId: "candidate-1",
    rallyIds: ["R001", "R002"],
  });
  tracking = removeSideSwitchMarker(tracking, "switch-candidate-1");
  assert.deepEqual(tracking.removedModelMarkerIds, ["switch-candidate-1"]);
  tracking = addSideSwitchMarker(tracking, 15, {
    id: "switch-candidate-1",
    origin: "model",
  });
  assert.equal(tracking.sideSwitchMarkers.length, 0);
});

test("disabled or suppressed rallies leave only visible markers in the scoring sequence", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", {
    id: "S001",
    origin: "model",
    rallyId: "R001",
  });
  tracking = addServeMarker(tracking, 10, "far", {
    id: "S002",
    origin: "model",
    rallyId: "R002",
  });
  tracking = addServeMarker(tracking, 20, "near", {
    id: "S003",
    origin: "model",
    rallyId: "R003",
  });

  const excludedRallyIds = new Set(["R002"]);
  const active = scoreTrackingOutsideExcludedRallies(
    tracking,
    excludedRallyIds,
  );
  assert.deepEqual(
    active.serveMarkers.map((marker) => marker.id),
    ["S001", "S003"],
  );

  const score = deriveScoreAt(active);
  assert.equal(score.team1Score, 1);
  assert.equal(score.team2Score, 0);
  assert.equal(score.ignoredPointCount, 0);
  assert.equal(score.points.length, active.serveMarkers.length - 1);
  assert.equal(tracking.serveMarkers.length, 3, "cached markers stay intact");

  assert.equal(
    scoreTrackingOutsideExcludedRallies(tracking, new Set()).serveMarkers,
    tracking.serveMarkers,
    "re-enabling the rally restores the original marker collection",
  );
});

test("score bounds use the next serve in dead time and leading padding", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 20, "far", { id: "S002" });
  tracking = addServeMarker(tracking, 30, "near", { id: "S003" });
  const ranges = [
    { keepStart: 3, coreStart: 5, coreEnd: 12, keepEnd: 14 },
    { keepStart: 18, coreStart: 20, coreEnd: 25, keepEnd: 27 },
  ];

  assert.equal(scoreBoundaryTimestamp(1, ranges, tracking), 5);
  assert.equal(scoreBoundaryTimestamp(4, ranges, tracking), 5);
  assert.equal(scoreBoundaryTimestamp(8, ranges, tracking), 8);
  assert.equal(scoreBoundaryTimestamp(13, ranges, tracking), 13);
  assert.equal(scoreBoundaryTimestamp(15, ranges, tracking), 20);
  assert.equal(scoreBoundaryTimestamp(40, ranges, tracking), 40);
});

test("merged rally fragments do not advance score inside an internal gap without a serve", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 20, "far", { id: "S002" });
  const ranges = [
    { keepStart: 3, coreStart: 5, coreEnd: 12, keepEnd: 14 },
    { keepStart: 16, coreStart: 20, coreEnd: 25, keepEnd: 27 },
  ];
  const merged = [{ start: 3, end: 27 }];

  assert.equal(scoreBoundaryTimestamp(13, ranges, tracking, merged), 13);
  assert.equal(scoreBoundaryTimestamp(15, ranges, tracking, merged), 15);
  assert.equal(scoreBoundaryTimestamp(19, ranges, tracking, merged), 19);
  assert.equal(scoreBoundaryTimestamp(20, ranges, tracking, merged), 20);
});

test("a serve marker inside merged padding activates the upcoming rally state", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 5, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 18, "far", { id: "S002" });
  tracking = addServeMarker(tracking, 30, "near", { id: "S003" });
  const ranges = [
    { keepStart: 3, coreStart: 5, coreEnd: 12, keepEnd: 14 },
    { keepStart: 16, coreStart: 20, coreEnd: 25, keepEnd: 27 },
  ];

  assert.equal(
    scoreBoundaryTimestamp(15, ranges, tracking, [{ start: 3, end: 27 }]),
    18,
  );
  assert.equal(
    scoreBoundaryTimestamp(19, ranges, tracking, [{ start: 3, end: 27 }]),
    19,
  );
});

test("first leading padding stops looking ahead after crossing its serve marker", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 2.5, "near", { id: "S001" });
  tracking = addServeMarker(tracking, 20, "far", { id: "S002" });
  const ranges = [{ keepStart: 3, coreStart: 5, coreEnd: 12, keepEnd: 14 }];
  const merged = [{ start: 3, end: 14 }];

  assert.equal(scoreBoundaryTimestamp(3, ranges, tracking, merged), 3);
  assert.equal(scoreBoundaryTimestamp(3.5, ranges, tracking, merged), 3.5);
  assert.equal(scoreBoundaryTimestamp(4.5, ranges, tracking, merged), 4.5);
});
