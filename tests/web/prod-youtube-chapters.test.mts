import assert from "node:assert/strict";
import test from "node:test";

import type {
  EditableCut,
  FinalCutInterval,
} from "../../prod/src/lib/cut-draft.ts";
import {
  addServeMarker,
  addSideSwitchMarker,
  createScoreTracking,
} from "../../prod/src/lib/score-tracking.ts";
import {
  buildYouTubeChapters,
  defaultYouTubeChapterOptions,
  formatYouTubeChapterTimestamp,
  type YouTubeChapterOptions,
  youtubeChaptersFilename,
  youtubeChaptersText,
} from "../../prod/src/lib/youtube-chapters.ts";

const intervals: FinalCutInterval[] = [
  { start: 10, end: 20, cutIds: ["R001"] },
  { start: 40, end: 50, cutIds: ["R002"] },
];

const cuts: EditableCut[] = [
  {
    id: "R001",
    coreStart: 12,
    coreEnd: 18,
    keepStart: 10,
    keepEnd: 20,
    confidence: 0.9,
    included: true,
    origin: "cached-label",
  },
  {
    id: "R002",
    coreStart: 42,
    coreEnd: 48,
    keepStart: 40,
    keepEnd: 50,
    confidence: 0.9,
    included: true,
    origin: "cached-label",
  },
];

const rallyOnly: YouTubeChapterOptions = {
  includeRallyNumber: true,
  includeServeNumber: false,
  includeScore: false,
  includeServingTeam: false,
  includeSideSwitches: false,
};

test("chapter defaults favor score and serving team when scoring is available", () => {
  assert.deepEqual(defaultYouTubeChapterOptions(true, true), {
    includeRallyNumber: false,
    includeServeNumber: false,
    includeScore: true,
    includeServingTeam: true,
    includeSideSwitches: true,
  });
  assert.deepEqual(defaultYouTubeChapterOptions(false, false), rallyOnly);
});

test("chapter timestamps follow the concatenated final MP4 timeline", () => {
  const chapters = buildYouTubeChapters({
    intervals,
    cuts,
    scoreTracking: null,
    options: rallyOnly,
  });

  assert.deepEqual(
    chapters.map(({ outputSeconds, title }) => ({ outputSeconds, title })),
    [
      { outputSeconds: 0, title: "Rally 1" },
      { outputSeconds: 10, title: "Rally 2" },
    ],
  );
  assert.equal(youtubeChaptersText(chapters), "0:00 Rally 1\n0:10 Rally 2");
});

test("score and serving-team titles use checked serve and team data", () => {
  let tracking = createScoreTracking();
  tracking = { ...tracking, team1Name: "Falcons", team2Name: "Wolves" };
  tracking = addServeMarker(tracking, 12, "near", {
    id: "S001",
    rallyId: "R001",
  });
  tracking = addServeMarker(tracking, 42, "far", {
    id: "S002",
    rallyId: "R002",
  });

  const chapters = buildYouTubeChapters({
    intervals,
    cuts,
    scoreTracking: tracking,
    options: {
      ...rallyOnly,
      includeRallyNumber: false,
      includeScore: true,
      includeServingTeam: true,
    },
  });

  assert.deepEqual(
    chapters.map((chapter) => chapter.title),
    ["0–0 - Falcons serving", "0–1 - Wolves serving"],
  );
});

test("a following ignored-point serve labels the preceding rally as a re-do", () => {
  let tracking = createScoreTracking();
  tracking = { ...tracking, team1Name: "Falcons", team2Name: "Wolves" };
  tracking = addServeMarker(tracking, 12, "near", {
    id: "S001",
    rallyId: "R001",
  });
  tracking = addServeMarker(tracking, 42, "far", {
    id: "S002",
    rallyId: "R002",
  });
  tracking = {
    ...tracking,
    serveMarkers: tracking.serveMarkers.map((marker) =>
      marker.id === "S002" ? { ...marker, ignorePreviousPoint: true } : marker,
    ),
  };

  const chapters = buildYouTubeChapters({
    intervals,
    cuts,
    scoreTracking: tracking,
    options: {
      ...rallyOnly,
      includeScore: true,
      includeServingTeam: true,
    },
  });

  assert.equal(chapters[0].title, "Rally 1 - 0–0 - Falcons serving - Re-do");
  assert.equal(chapters[1].title, "Rally 2 - 0–0 - Wolves serving");
});

test("manually added side switches become chapters without a project-creation flag", () => {
  let tracking = createScoreTracking();
  tracking = addSideSwitchMarker(tracking, 15, "X001");
  tracking = addSideSwitchMarker(tracking, 30, "X002");
  tracking = addSideSwitchMarker(tracking, 45, "X003");

  const chapters = buildYouTubeChapters({
    intervals,
    cuts,
    scoreTracking: tracking,
    options: { ...rallyOnly, includeSideSwitches: true },
  });

  assert.deepEqual(
    chapters.map(({ outputSeconds, title }) => [outputSeconds, title]),
    [
      [0, "Rally 1"],
      [5, "Side switch 1"],
      [10, "Rally 2 / Side switch 2"],
      [15, "Side switch 3"],
    ],
  );
});

test("chapter helpers support long videos and safe text filenames", () => {
  assert.equal(formatYouTubeChapterTimestamp(3661.9), "1:01:01");
  assert.equal(
    youtubeChaptersFilename("My match (final).mov"),
    "My-match-final-youtube-chapters.txt",
  );
});
