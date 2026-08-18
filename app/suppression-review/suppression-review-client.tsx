"use client";

import Link from "next/link";
import { useMemo, useRef, useState } from "react";

import { Brand } from "@/components/brand";
import { formatTime } from "@/lib/edit-list";

import styles from "./suppression-review.module.css";
import type {
  AffectedRally,
  MetricTriplet,
  ReviewInterval,
  ReviewTrack,
  SuppressionReviewDataset,
  SuppressionReviewVideo,
} from "./types";

type Viewport = { start: number; end: number };
type IntervalKind = "raw" | "padding" | "joined";
type VisibleInterval = ReviewInterval & { affected: boolean };

const TRACKS: Array<{
  key: keyof SuppressionReviewVideo["tracks"];
  label: string;
  detail: string;
  tone: string;
}> = [
  {
    key: "affectedRallies",
    label: "Affected rally",
    detail: "review target",
    tone: "affected",
  },
  {
    key: "human",
    label: "Human labels",
    detail: "all reviewed rallies",
    tone: "human",
  },
  {
    key: "previousProduction",
    label: "Previous prod.",
    detail: "+2s, joins <3s",
    tone: "previous",
  },
  {
    key: "allLabelsV2",
    label: "All-labels v2",
    detail: "+2s, joins <3s",
    tone: "v2",
  },
  {
    key: "ensemble",
    label: "Production union",
    detail: "before veto",
    tone: "ensemble",
  },
  {
    key: "suppressionApplied",
    label: "Veto rail",
    detail: "one-model export only",
    tone: "veto",
  },
  {
    key: "ensembleSuppressed",
    label: "Union + veto",
    detail: "inspection candidate",
    tone: "candidate",
  },
];

function compactSeconds(value: number): string {
  return `${value.toFixed(value >= 10 ? 1 : 2)}s`;
}

function metric(value: number): string {
  return value.toFixed(4);
}

function signedMetric(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
}

function provenanceLabel(video: SuppressionReviewVideo): string {
  if (video.provenance === "export-feedback") {
    return `Feedback · ${video.feedbackPartition ?? "unknown split"}`;
  }
  return video.provenance === "training-dataset"
    ? "Training dataset"
    : "Evaluation / validation / test";
}

const METRIC_DEFINITIONS = {
  precision:
    "Overlap duration divided by predicted duration. Higher means less unnecessary footage.",
  recall:
    "Overlap duration divided by reviewed human duration. Higher means more reviewed play is retained.",
  f1: "Harmonic mean of precision and recall for this same core or padded comparison.",
} as const;

function MetricMovement({
  label,
  detail,
  production,
  candidate,
}: {
  label: string;
  detail: string;
  production: MetricTriplet;
  candidate: MetricTriplet;
}) {
  const entries = [
    ["precision", "Precision"],
    ["recall", "Recall"],
    ["f1", "F1"],
  ] as const;
  return (
    <div className={styles.movementCard}>
      <div>
        <strong>{label}</strong>
        <small>{detail}</small>
      </div>
      <dl>
        {entries.map(([key, name]) => (
          <div title={METRIC_DEFINITIONS[key]} key={key}>
            <dt>{name}</dt>
            <dd>
              <span>{metric(production[key])}</span>
              <b aria-hidden="true">→</b>
              <strong>{metric(candidate[key])}</strong>
            </dd>
            <small
              data-direction={
                candidate[key] - production[key] < 0 ? "down" : "up"
              }
            >
              {signedMetric(candidate[key] - production[key])}
            </small>
          </div>
        ))}
      </dl>
    </div>
  );
}

function overlaps(left: ReviewInterval, right: ReviewInterval): boolean {
  return left.start < right.end && right.start < left.end;
}

function clipInterval(
  interval: ReviewInterval,
  viewport: Viewport,
): ReviewInterval | null {
  const start = Math.max(interval.start, viewport.start);
  const end = Math.min(interval.end, viewport.end);
  return end > start ? { start, end } : null;
}

function splitByFocus(
  interval: ReviewInterval,
  focusRanges: ReviewInterval[],
): VisibleInterval[] {
  const cuts = new Set([interval.start, interval.end]);
  for (const focus of focusRanges) {
    if (!overlaps(interval, focus)) continue;
    cuts.add(Math.max(interval.start, focus.start));
    cuts.add(Math.min(interval.end, focus.end));
  }
  const points = [...cuts].sort((left, right) => left - right);
  return points.slice(0, -1).map((start, index) => {
    const end = points[index + 1];
    const midpoint = (start + end) / 2;
    return {
      ...interval,
      start,
      end,
      affected: focusRanges.some(
        (focus) => midpoint >= focus.start && midpoint <= focus.end,
      ),
    };
  });
}

function visibleIntervals(
  intervals: ReviewInterval[],
  viewport: Viewport,
  focusRanges: ReviewInterval[],
): VisibleInterval[] {
  return intervals.flatMap((interval) => {
    const clipped = clipInterval(interval, viewport);
    return clipped ? splitByFocus(clipped, focusRanges) : [];
  });
}

function intervalStyle(interval: ReviewInterval, viewport: Viewport) {
  const duration = viewport.end - viewport.start;
  return {
    left: `${((interval.start - viewport.start) / duration) * 100}%`,
    width: `${((interval.end - interval.start) / duration) * 100}%`,
  };
}

function confidence(value: number): string {
  return value.toFixed(3);
}

function rawPredictionsNearRally(
  track: ReviewTrack,
  rally: AffectedRally,
): ReviewInterval[] {
  const context = { start: Math.max(0, rally.start - 2), end: rally.end + 2 };
  return track.raw.filter(
    (interval) =>
      typeof interval.confidence === "number" && overlaps(interval, context),
  );
}

function focusViewport(
  video: SuppressionReviewVideo,
  rally: AffectedRally,
): Viewport {
  const related = video.focusRanges.find((range) => overlaps(range, rally));
  const contextual = {
    start: Math.max(0, (related?.start ?? rally.start - 2) - 6),
    end: Math.min(video.duration, (related?.end ?? rally.end + 2) + 6),
  };
  if (contextual.end - contextual.start >= 18) return contextual;
  const middle = (contextual.start + contextual.end) / 2;
  return {
    start: Math.max(0, middle - 9),
    end: Math.min(video.duration, middle + 9),
  };
}

function TimelineRail({
  track,
  tone,
  viewport,
  focusRanges,
  currentTime,
  onSeek,
}: {
  track: ReviewTrack;
  tone: string;
  viewport: Viewport;
  focusRanges: ReviewInterval[];
  currentTime: number;
  onSeek: (time: number) => void;
}) {
  const layers: Array<[IntervalKind, ReviewInterval[]]> = [
    ["joined", track.joinedGaps],
    ["padding", track.padding],
    ["raw", track.raw],
  ];
  return (
    <div
      className={styles.rail}
      onClick={(event) => {
        const bounds = event.currentTarget.getBoundingClientRect();
        const ratio = Math.max(
          0,
          Math.min(1, (event.clientX - bounds.left) / bounds.width),
        );
        onSeek(viewport.start + ratio * (viewport.end - viewport.start));
      }}
      role="presentation"
    >
      {focusRanges.map((focus) => {
        const clipped = clipInterval(focus, viewport);
        return clipped ? (
          <span
            className={styles.focusBand}
            style={intervalStyle(clipped, viewport)}
            key={`focus-${focus.start}-${focus.end}`}
          />
        ) : null;
      })}
      {layers.flatMap(([kind, intervals]) =>
        visibleIntervals(intervals, viewport, focusRanges).map((interval) => (
          <button
            type="button"
            className={styles.interval}
            data-kind={kind}
            data-tone={tone}
            data-affected={interval.affected ? "true" : "false"}
            style={intervalStyle(interval, viewport)}
            onClick={(event) => {
              event.stopPropagation();
              onSeek(interval.start);
            }}
            title={`${kind}: ${formatTime(interval.start)}–${formatTime(interval.end)}${typeof interval.confidence === "number" ? ` · score ${confidence(interval.confidence)}` : ""}${interval.affected ? " · affected context" : " · outside affected context"}`}
            aria-label={`Seek to ${formatTime(interval.start)}`}
            key={`${kind}-${interval.start}-${interval.end}-${interval.affected}`}
          />
        )),
      )}
      {currentTime >= viewport.start && currentTime <= viewport.end && (
        <span
          className={styles.playhead}
          style={{
            left: `${((currentTime - viewport.start) / (viewport.end - viewport.start)) * 100}%`,
          }}
        />
      )}
    </div>
  );
}

function TimeAxis({ viewport }: { viewport: Viewport }) {
  return (
    <div className={styles.timeAxis}>
      {Array.from({ length: 6 }, (_, index) => {
        const time =
          viewport.start + (viewport.end - viewport.start) * (index / 5);
        return <span key={time}>{formatTime(time)}</span>;
      })}
    </div>
  );
}

function Overview({
  video,
  currentTime,
  onSeek,
}: {
  video: SuppressionReviewVideo;
  currentTime: number;
  onSeek: (time: number) => void;
}) {
  const viewport = { start: 0, end: video.duration };
  return (
    <div className={styles.overview}>
      <div className={styles.overviewLabel}>Affected map</div>
      <div
        className={styles.overviewRail}
        onClick={(event) => {
          const bounds = event.currentTarget.getBoundingClientRect();
          onSeek(
            ((event.clientX - bounds.left) / bounds.width) * video.duration,
          );
        }}
        role="presentation"
      >
        {video.focusRanges.map((range) => (
          <span
            className={styles.overviewFocus}
            style={intervalStyle(range, viewport)}
            key={`${range.start}-${range.end}`}
          />
        ))}
        {video.affectedRallies.map((rally) => (
          <span
            className={styles.overviewMiss}
            data-complete={rally.completeMiss ? "true" : "false"}
            style={intervalStyle(rally, viewport)}
            key={rally.rallyNumber}
          />
        ))}
        <span
          className={styles.overviewPlayhead}
          style={{ left: `${(currentTime / video.duration) * 100}%` }}
        />
      </div>
      <span>{formatTime(video.duration)}</span>
    </div>
  );
}

function VideoReviewCard({
  video,
  index,
}: {
  video: SuppressionReviewVideo;
  index: number;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [expanded, setExpanded] = useState(index === 0);
  const [currentTime, setCurrentTime] = useState(
    video.affectedRallies[0]?.start ?? 0,
  );
  const [selectedRallyNumber, setSelectedRallyNumber] = useState(
    video.affectedRallies[0]?.rallyNumber ?? 1,
  );
  const [fullTimeline, setFullTimeline] = useState(false);
  const selectedRally =
    video.affectedRallies.find(
      (rally) => rally.rallyNumber === selectedRallyNumber,
    ) ?? video.affectedRallies[0];
  const viewport = fullTimeline
    ? { start: 0, end: video.duration }
    : focusViewport(video, selectedRally);
  const coverage =
    selectedRally.duration > 0
      ? selectedRally.candidateCoveredSeconds / selectedRally.duration
      : 0;
  const confidenceGroups = [
    {
      label: "Previous prod.",
      intervals: rawPredictionsNearRally(
        video.tracks.previousProduction,
        selectedRally,
      ),
    },
    {
      label: "All-labels v2",
      intervals: rawPredictionsNearRally(
        video.tracks.allLabelsV2,
        selectedRally,
      ),
    },
    {
      label: "Veto",
      intervals: rawPredictionsNearRally(
        video.tracks.suppressionApplied,
        selectedRally,
      ),
    },
  ];

  function seek(time: number) {
    const bounded = Math.max(0, Math.min(video.duration, time));
    if (videoRef.current) videoRef.current.currentTime = bounded;
    setCurrentTime(bounded);
  }

  function selectRally(rally: AffectedRally) {
    setSelectedRallyNumber(rally.rallyNumber);
    setFullTimeline(false);
    seek(Math.max(0, rally.start - 2));
  }

  return (
    <article
      className={styles.videoCard}
      data-severity={video.completeMisses > 0 ? "complete" : "partial"}
    >
      <header className={styles.cardHeader}>
        <span className={styles.cardIndex}>
          {String(index + 1).padStart(2, "0")}
        </span>
        <div>
          <p>
            {video.environment} · {provenanceLabel(video)}
          </p>
          <h2>{video.file}</h2>
        </div>
        <div className={styles.cardActions}>
          <div className={styles.cardBadges}>
            {video.completeMisses > 0 && <b>{video.completeMisses} complete</b>}
            <span>{video.partialMisses} partial</span>
            <span>{compactSeconds(video.lostCoreSeconds)} lost</span>
          </div>
          <button type="button" onClick={() => setExpanded((value) => !value)}>
            {expanded ? "Close inspection" : "Open inspection"}
          </button>
        </div>
      </header>

      {expanded && (
        <div className={styles.inspectionGrid}>
          <div className={styles.playerColumn}>
            <video
              ref={videoRef}
              controls
              playsInline
              preload="metadata"
              src={video.videoUrl}
              onTimeUpdate={(event) =>
                setCurrentTime(event.currentTarget.currentTime)
              }
              onLoadedMetadata={(event) => {
                if (event.currentTarget.currentTime === 0) {
                  event.currentTarget.currentTime = Math.max(
                    0,
                    selectedRally.start - 2,
                  );
                }
              }}
              aria-label={`Video review for ${video.file}`}
            >
              Your browser does not support video playback.
            </video>
            <div className={styles.rallyPicker}>
              {video.affectedRallies.map((rally) => (
                <button
                  type="button"
                  data-active={
                    rally.rallyNumber === selectedRally.rallyNumber
                      ? "true"
                      : "false"
                  }
                  data-complete={rally.completeMiss ? "true" : "false"}
                  onClick={() => selectRally(rally)}
                  key={rally.rallyNumber}
                >
                  <strong>R{rally.rallyNumber}</strong>
                  <span>
                    {formatTime(rally.start)}–{formatTime(rally.end)}
                  </span>
                  <small>−{compactSeconds(rally.lostCoreSeconds)}</small>
                </button>
              ))}
            </div>
          </div>

          <div className={styles.timelineColumn}>
            <div className={styles.selectedSummary}>
              <div>
                <span>
                  {selectedRally.completeMiss
                    ? "Complete miss"
                    : "Partial miss"}
                </span>
                <strong>Rally {selectedRally.rallyNumber}</strong>
              </div>
              <dl>
                <div>
                  <dt>Human core</dt>
                  <dd>{compactSeconds(selectedRally.duration)}</dd>
                </div>
                <div>
                  <dt>Coverage lost</dt>
                  <dd>−{compactSeconds(selectedRally.lostCoreSeconds)}</dd>
                </div>
                <div>
                  <dt>Still covered</dt>
                  <dd>{(coverage * 100).toFixed(1)}%</dd>
                </div>
              </dl>
              <button
                type="button"
                onClick={() => setFullTimeline((value) => !value)}
              >
                {fullTimeline ? "Focus selected rally" : "Show full video"}
              </button>
            </div>
            <Overview video={video} currentTime={currentTime} onSeek={seek} />
            <div className={styles.confidencePanel}>
              <div>
                <strong>Scores near selected rally</strong>
                <small>uncalibrated interval scores</small>
              </div>
              <div className={styles.confidenceGroups}>
                {confidenceGroups.map((group) => (
                  <div className={styles.confidenceGroup} key={group.label}>
                    <span>{group.label}</span>
                    {group.intervals.length > 0 ? (
                      group.intervals.map((interval) => (
                        <button
                          type="button"
                          onClick={() => seek(interval.start)}
                          title={`${group.label}: ${formatTime(interval.start)}–${formatTime(interval.end)}`}
                          key={`${interval.start}-${interval.end}`}
                        >
                          {confidence(interval.confidence ?? 0)}
                          <small>
                            {formatTime(interval.start)}–
                            {formatTime(interval.end)}
                          </small>
                        </button>
                      ))
                    ) : (
                      <em>no raw interval</em>
                    )}
                  </div>
                ))}
              </div>
              <p>
                Union and post-veto rails are derived ranges, so they have no
                standalone score. Hover any scored raw bar for its value.
              </p>
            </div>
            <div className={styles.timelineHeader}>
              <span>Rail</span>
              <TimeAxis viewport={viewport} />
            </div>
            <div className={styles.tracks}>
              {TRACKS.map((definition) => (
                <div className={styles.track} key={definition.key}>
                  <div className={styles.trackLabel}>
                    <strong>{definition.label}</strong>
                    <small>{definition.detail}</small>
                  </div>
                  <TimelineRail
                    track={video.tracks[definition.key]}
                    tone={definition.tone}
                    viewport={viewport}
                    focusRanges={video.focusRanges}
                    currentTime={currentTime}
                    onSeek={seek}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </article>
  );
}

export function SuppressionReviewClient({
  dataset,
}: {
  dataset: SuppressionReviewDataset;
}) {
  const sections = useMemo(
    () => ({
      complete: dataset.videos.filter((video) => video.completeMisses > 0),
      partial: dataset.videos.filter((video) => video.completeMisses === 0),
    }),
    [dataset.videos],
  );
  const precisionDelta =
    dataset.summary.candidate.P_pad - dataset.summary.production.P_pad;
  const recallDelta =
    dataset.summary.candidate.R_core - dataset.summary.production.R_core;
  const retrained = dataset.modelVariantId?.startsWith(
    "overlap-exclusion-retrained",
  );
  const modelVariantLabel =
    dataset.modelVariantLabel ?? "Original suppression specialist";

  return (
    <main className={styles.shell}>
      <nav className={styles.topbar}>
        <Brand className={styles.brand} label="Suppression review" priority />
        <div>
          <Link href="/">Model review</Link>
          <Link href="/model-feedback">Feedback</Link>
        </div>
      </nav>

      <header className={styles.hero}>
        <div>
          <p className={styles.eyebrow}>
            {dataset.policyLabel} · {modelVariantLabel} · visual audit
          </p>
          <h1>
            See exactly what <em>would disappear.</em>
          </h1>
          <p className={styles.intro}>
            {dataset.strategy} Every rail uses 2-second symmetric padding,
            ignored-range subtraction, and joins positive gaps strictly below 3
            seconds.
          </p>
        </div>
        <div className={styles.metricPair}>
          <span>P_pad</span>
          <strong>{metric(dataset.summary.candidate.P_pad)}</strong>
          <small>+{metric(precisionDelta)}</small>
          <span>R_core</span>
          <strong>{metric(dataset.summary.candidate.R_core)}</strong>
          <small>{metric(recallDelta)}</small>
        </div>
      </header>

      <section className={styles.policySwitch} aria-label="Suppression policy">
        <div>
          <span>Compare review version</span>
          <strong>
            {dataset.policyLabel} · {modelVariantLabel}
          </strong>
          <p>{dataset.strategy}</p>
        </div>
        <nav>
          <Link
            data-active={dataset.policyId === "pointwise" ? "true" : "false"}
            href="/suppression-review"
          >
            Pointwise overlap
            <small>Only exact overlapping time is protected</small>
          </Link>
          <Link
            data-active={
              dataset.policyId === "any-overlap" && !retrained
                ? "true"
                : "false"
            }
            href="/suppression-review/any-overlap"
          >
            Any overlap protects export span
            <small>After padding and joins, protects the full component</small>
          </Link>
          <Link
            data-active={
              dataset.modelVariantId === "overlap-exclusion-retrained"
                ? "true"
                : "false"
            }
            href="/suppression-review/any-overlap/retrained"
          >
            Export overlap · retrained specialist
            <small>Excludes ambiguous overlapping feedback targets</small>
          </Link>
          <Link
            data-active={
              dataset.policyId === "any-overlap-raw" ||
              dataset.modelVariantId?.startsWith(
                "overlap-exclusion-retrained-pad-",
              )
                ? "true"
                : "false"
            }
            href="/suppression-review/tuning"
          >
            Agreement tuning
            <small>Compare padding and join grouping breakpoints</small>
          </Link>
        </nav>
      </section>

      <section className={styles.summaryStrip} aria-label="Review summary">
        <div>
          <span>Videos</span>
          <strong>{dataset.summary.videos}</strong>
        </div>
        <div>
          <span>Affected rallies</span>
          <strong>{dataset.summary.affectedRallies}</strong>
        </div>
        <div data-tone="danger">
          <span>Complete misses</span>
          <strong>{dataset.summary.completeMisses}</strong>
        </div>
        <div data-tone="warning">
          <span>Partial misses</span>
          <strong>{dataset.summary.partialMisses}</strong>
        </div>
        <div>
          <span>Core coverage lost</span>
          <strong>{compactSeconds(dataset.summary.lostCoreSeconds)}</strong>
        </div>
      </section>

      <section className={styles.outcomePanel} aria-label="Policy outcome">
        <div className={styles.outcomeLead}>
          <div title="Reduction in the canonical model export after 2-second symmetric padding, ignored-range subtraction, and joining positive gaps strictly below 3 seconds.">
            <span>Export time saved</span>
            <strong>
              {compactSeconds(dataset.summary.exportTimeSavedSeconds)}
            </strong>
            <small>
              {compactSeconds(
                dataset.summary.production.paddedModelExportSeconds,
              )}{" "}
              →{" "}
              {compactSeconds(
                dataset.summary.candidate.paddedModelExportSeconds,
              )}
            </small>
          </div>
          <div title="Production prediction ranges fully deleted by this policy that have zero overlap with the padded human export target after ignored-range subtraction.">
            <span>Correct FP rallies removed</span>
            <strong>
              {dataset.summary.correctlyRemovedFalsePositivePredictions}
            </strong>
            <small>
              Fully deleted · no padded-human overlap ·{" "}
              {compactSeconds(dataset.summary.correctlyRemovedRawSeconds)} raw
            </small>
          </div>
        </div>
        <div className={styles.movementGrid}>
          <MetricMovement
            label="Core movement"
            detail="Unpadded model vs. human core"
            production={dataset.summary.production.core}
            candidate={dataset.summary.candidate.core}
          />
          <MetricMovement
            label="Padded movement"
            detail="Both sides +2s, then joins <3s"
            production={dataset.summary.production.padded}
            candidate={dataset.summary.candidate.padded}
          />
        </div>
      </section>

      <section className={styles.legend} aria-label="Timeline legend">
        <span data-kind="raw">Raw prediction / core</span>
        <span data-kind="padding">2-second padding</span>
        <span data-kind="joined">Joined &lt;3s gap</span>
        <span data-kind="muted">Outside affected context</span>
        <span data-kind="focus">Affected context</span>
        <span data-kind="score">Scored raw bars show confidence on hover</span>
      </section>

      <section className={styles.reviewSection}>
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.eyebrow}>Inspect first</p>
            <h2>Videos with complete misses</h2>
          </div>
          <strong>
            {sections.complete.length}{" "}
            {sections.complete.length === 1 ? "video" : "videos"} ·{" "}
            {dataset.summary.completeMisses} rallies
          </strong>
        </div>
        <div className={styles.cardList}>
          {sections.complete.map((video, index) => (
            <VideoReviewCard
              video={video}
              index={index}
              key={video.recordingId}
            />
          ))}
        </div>
      </section>

      <section className={styles.reviewSection}>
        <div className={styles.sectionHeading}>
          <div>
            <p className={styles.eyebrow}>Coverage clipped</p>
            <h2>Videos with partial misses only</h2>
          </div>
          <strong>
            {sections.partial.length}{" "}
            {sections.partial.length === 1 ? "video" : "videos"} ·{" "}
            {dataset.summary.partialMisses} rallies
          </strong>
        </div>
        <div className={styles.cardList}>
          {sections.partial.map((video, index) => (
            <VideoReviewCard
              video={video}
              index={sections.complete.length + index}
              key={video.recordingId}
            />
          ))}
        </div>
      </section>

      <footer className={styles.footer}>
        <p>
          Experiment {dataset.experiment} · Generated from reviewed labels and
          exact production component inference.
        </p>
        <Link href="/">Return to model review</Link>
      </footer>
    </main>
  );
}
