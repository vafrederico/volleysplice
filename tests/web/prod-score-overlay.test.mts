import assert from "node:assert/strict";
import test from "node:test";

import {
  drawScoreOverlay,
  type ScoreOverlayCanvasContext,
  formatOverlayScore,
  prepareScoreOverlay,
  scoreOverlayLayout,
  scoreOverlaySnapshot,
  scorePointTimelineLayout,
  scorePointTimelineSnapshot,
  visibleScorePointTimelineEntries,
} from "../../prod/src/lib/score-overlay.ts";
import {
  addServeMarker,
  createScoreTracking,
  setPreviousPointIgnored,
} from "../../prod/src/lib/score-tracking.ts";

function trackingFixture() {
  let tracking = createScoreTracking();
  tracking = {
    ...tracking,
    team1Name: "Falcons",
    team2Name: "Waves",
  };
  tracking = addServeMarker(tracking, 2, "near", { id: "S1", rallyId: "R1" });
  tracking = addServeMarker(tracking, 8, "near", { id: "S2", rallyId: "R2" });
  tracking = addServeMarker(tracking, 14, "far", { id: "S3", rallyId: "R3" });
  tracking = addServeMarker(tracking, 20, "near", { id: "S4", rallyId: "R4" });
  return tracking;
}

test("score rendering follows timeline alpha only when fading, even with the timeline hidden", () => {
  const prepared = prepareScoreOverlay({ scoreTracking: trackingFixture(), fadeScoreOverlay: true });
  assert.equal(prepareScoreOverlay({ scoreTracking: trackingFixture() }).fadeScoreOverlay, false);
  const stack: number[] = [];
  const scoreAlphas: number[] = [];
  const pointAlphas: number[] = [];
  const target = {
    globalAlpha: 1,
    save() { stack.push(this.globalAlpha); },
    restore() { this.globalAlpha = stack.pop()!; },
    measureText() { return { width: 80 }; },
    fillRect() { scoreAlphas.push(this.globalAlpha); },
    fill() { pointAlphas.push(this.globalAlpha); },
  };
  const context = new Proxy(target, {
    get(object, key) { return key in object ? Reflect.get(object, key) : () => {}; },
  }) as unknown as ScoreOverlayCanvasContext;
  for (const fade of [false, true]) {
    for (const renderTimeline of [false, true]) {
      for (const timestamp of [8, 8.125, 8.25, 10.25, 10.425, 10.6]) {
        scoreAlphas.length = 0;
        pointAlphas.length = 0;
        const timeline = scorePointTimelineSnapshot(prepared, timestamp);
        drawScoreOverlay(context, 1920, 1080, scoreOverlaySnapshot(prepared, timestamp), timeline, renderTimeline, fade);
        assert.deepEqual(scoreAlphas, Array(4).fill(fade ? timeline.opacity : 1));
        assert.ok(pointAlphas.every((alpha) => alpha === timeline.opacity));
        assert.equal(pointAlphas.length > 0, renderTimeline && timeline.opacity > 0);
        assert.equal(context.globalAlpha, 1);
      }
    }
  }
});

test("overlay scores are always padded to at least two digits", () => {
  assert.equal(formatOverlayScore(0), "00");
  assert.equal(formatOverlayScore(7), "07");
  assert.equal(formatOverlayScore(21), "21");
});

test("overlay snapshot follows source-timeline score history", () => {
  const prepared = prepareScoreOverlay({ scoreTracking: trackingFixture() });
  assert.equal(prepared.renderPointTimeline, true);
  assert.equal(scoreOverlaySnapshot(prepared, 1).servingTeamId, "team-1");
  assert.deepEqual(scoreOverlaySnapshot(prepared, 2), {
    team1Name: "Falcons",
    team1Score: 0,
    team1ScoreLabel: "00",
    team2Name: "Waves",
    team2Score: 0,
    team2ScoreLabel: "00",
    servingTeamId: "team-1",
  });
  assert.deepEqual(
    {
      team1: scoreOverlaySnapshot(prepared, 14).team1ScoreLabel,
      team2: scoreOverlaySnapshot(prepared, 14).team2ScoreLabel,
    },
    { team1: "01", team2: "01" },
  );
});

test("point timeline rendering can be disabled independently of the scoreboard", () => {
  const prepared = prepareScoreOverlay({
    scoreTracking: trackingFixture(),
    renderPointTimeline: false,
  });

  assert.equal(prepared.renderPointTimeline, false);
  assert.equal(scoreOverlaySnapshot(prepared, 2).team1ScoreLabel, "00");
});

test("overlay removes ignored, suppressed, and replayed points", () => {
  let tracking = trackingFixture();
  tracking = setPreviousPointIgnored(tracking, "S4", true);
  const prepared = prepareScoreOverlay({
    scoreTracking: tracking,
    excludedRallyIds: ["R2"],
    ignoredIntervals: [{ start: 13, end: 15 }],
  });
  const snapshot = scoreOverlaySnapshot(prepared, 30);
  assert.equal(snapshot.team1ScoreLabel, "00");
  assert.equal(snapshot.team2ScoreLabel, "00");
  assert.deepEqual(scorePointTimelineSnapshot(prepared, 30).points, []);
});

test("overlay uses the next serve during dead time and leading padding", () => {
  const prepared = prepareScoreOverlay({
    scoreTracking: trackingFixture(),
    rallyRanges: [
      { coreStart: 2, coreEnd: 5, keepStart: 1, keepEnd: 6 },
      { coreStart: 8, coreEnd: 11, keepStart: 7, keepEnd: 12 },
      { coreStart: 14, coreEnd: 17, keepStart: 13, keepEnd: 18 },
    ],
  });
  assert.equal(scoreOverlaySnapshot(prepared, 6.5).team1ScoreLabel, "01");
  assert.equal(scoreOverlaySnapshot(prepared, 7.5).team1ScoreLabel, "01");
  assert.equal(scoreOverlaySnapshot(prepared, 12.5).team2ScoreLabel, "01");
  assert.equal(scoreOverlaySnapshot(prepared, 7.5).servingTeamId, "team-1");
  assert.equal(scoreOverlaySnapshot(prepared, 12.5).servingTeamId, "team-2");
  assert.equal(scoreOverlaySnapshot(prepared, 13.5).servingTeamId, "team-2");
  assert.equal(scoreOverlaySnapshot(prepared, 14).servingTeamId, "team-2");
});

test("point timeline reveals awarded points at leading padding, holds, and fades", () => {
  const prepared = prepareScoreOverlay({
    scoreTracking: trackingFixture(),
    rallyRanges: [
      { coreStart: 2, coreEnd: 5, keepStart: 1, keepEnd: 6 },
      { coreStart: 8, coreEnd: 11, keepStart: 7, keepEnd: 12 },
      { coreStart: 14, coreEnd: 17, keepStart: 13, keepEnd: 18 },
    ],
  });

  assert.deepEqual(scorePointTimelineSnapshot(prepared, 7).points, [
    {
      serveMarkerId: "S2",
      winnerTeamId: "team-1",
      teamPointNumber: 1,
    },
  ]);
  assert.equal(scorePointTimelineSnapshot(prepared, 7).opacity, 0);
  assert.equal(scorePointTimelineSnapshot(prepared, 7.125).opacity, 0.5);
  assert.equal(scorePointTimelineSnapshot(prepared, 7.25).opacity, 1);
  assert.equal(scorePointTimelineSnapshot(prepared, 9.25).opacity, 1);
  assert.ok(
    Math.abs(scorePointTimelineSnapshot(prepared, 9.425).opacity - 0.5) < 1e-9,
  );
  assert.equal(scorePointTimelineSnapshot(prepared, 9.6).opacity, 0);

  assert.deepEqual(scorePointTimelineSnapshot(prepared, 13.25).points, [
    { serveMarkerId: "S2", winnerTeamId: "team-1", teamPointNumber: 1 },
    { serveMarkerId: "S3", winnerTeamId: "team-2", teamPointNumber: 1 },
  ]);
});

test("overlay keeps one score state across merged fragments without an internal serve", () => {
  const prepared = prepareScoreOverlay({
    scoreTracking: trackingFixture(),
    rallyRanges: [
      { coreStart: 2, coreEnd: 5, keepStart: 1, keepEnd: 6 },
      { coreStart: 8, coreEnd: 11, keepStart: 7, keepEnd: 12 },
    ],
    mergedRanges: [{ start: 1, end: 12 }],
  });

  const snapshot = scoreOverlaySnapshot(prepared, 6.5);
  assert.equal(snapshot.team1ScoreLabel, "00");
  assert.equal(snapshot.team2ScoreLabel, "00");
  assert.equal(snapshot.servingTeamId, "team-1");
});

test("overlay does not award the first point after crossing a serve in leading padding", () => {
  let tracking = createScoreTracking();
  tracking = addServeMarker(tracking, 2.5, "near", { id: "S1" });
  tracking = addServeMarker(tracking, 20, "far", { id: "S2" });
  const prepared = prepareScoreOverlay({
    scoreTracking: tracking,
    rallyRanges: [{ coreStart: 5, coreEnd: 12, keepStart: 3, keepEnd: 14 }],
    mergedRanges: [{ start: 3, end: 14 }],
  });

  const snapshot = scoreOverlaySnapshot(prepared, 4.5);
  assert.equal(snapshot.team1ScoreLabel, "00");
  assert.equal(snapshot.team2ScoreLabel, "00");
  assert.equal(snapshot.servingTeamId, "team-1");
});

test("overlay layout grows for team names, stays bounded, and gives each score equal width", () => {
  const snapshot = scoreOverlaySnapshot(
    prepareScoreOverlay({ scoreTracking: trackingFixture() }),
    30,
  );
  const layout = scoreOverlayLayout(
    {
      measureText: (text: string) =>
        ({ width: text.length * 10 }) as TextMetrics,
    },
    1920,
    1080,
    snapshot,
  );
  assert.equal(layout.height, 69);
  assert.equal(layout.borderWidth, 2);
  assert.equal(layout.scoreWidth, 90);
  assert.ok(layout.width < 1920 / 2);
  assert.ok(layout.radius > 0);

  const longNameLayout = scoreOverlayLayout(
    {
      measureText: (text: string) =>
        ({ width: text.length * 18 }) as TextMetrics,
    },
    1920,
    1080,
    {
      ...snapshot,
      team1Name: "Very Long Home Team Name",
      team2Name: "International Volleyball Club",
    },
  );
  assert.ok(longNameLayout.team1Width > layout.team1Width);
  assert.ok(longNameLayout.team2Width > layout.team2Width);
  assert.ok(longNameLayout.width <= Math.floor(1920 * 0.96));

  const longPoints = Array.from({ length: 38 }, (_, index) => ({
    serveMarkerId: `S${index + 1}`,
    winnerTeamId:
      index % 2 === 0 ? ("team-1" as const) : ("team-2" as const),
    teamPointNumber: Math.floor(index / 2) + 1,
  }));
  const visiblePoints = visibleScorePointTimelineEntries(
    1920,
    layout,
    longPoints,
  );
  const timelineLayout = scorePointTimelineLayout(
    1920,
    1080,
    layout,
    visiblePoints.length,
  );
  assert.equal(timelineLayout.startX, layout.width);
  assert.ok(timelineLayout.team1CenterY < timelineLayout.team2CenterY);
  assert.ok(visiblePoints.length < longPoints.length);
  assert.equal(
    visiblePoints.at(0)?.serveMarkerId,
    `S${39 - visiblePoints.length}`,
  );
  assert.equal(visiblePoints.at(-1)?.serveMarkerId, "S38");
  assert.ok(
    timelineLayout.columnSpacing * visiblePoints.length <= 1920 - layout.width,
  );
  assert.ok(timelineLayout.circleRadius > 0);
  assert.ok(timelineLayout.lineWidth >= 2);

  const shortTimelineLayout = scorePointTimelineLayout(1920, 1080, layout, 8);
  assert.equal(shortTimelineLayout.columnSpacing, layout.height * 0.58);
  assert.ok(shortTimelineLayout.circleRadius > layout.height * 0.17);
});
